//! Canonical `detect_value_encoding` and `values_to_raw_bytes` implementations.
//!
//! Until this module existed, three crates (`pyscx/anndata.rs`,
//! `scx-cli/convert/dtype.rs`, `scx-mtx/convert.rs`) each carried their own
//! structurally-identical copy of the detect + encode logic. Per finding
//! H11, all three now delegate here.
//!
//! `values_to_raw_bytes` is a thin wrapper around
//! [`ValueEncoding::encode_f32_batch`](crate::dispatch::ValueEncoding::encode_f32_batch),
//! which holds the range-checked per-encoding serialiser. Future tweaks to
//! per-encoding byte layout happen there; this module owns only the
//! integer-detection policy.
//!
//! Conventions:
//! - Integer detection uses `v.is_finite() && v >= 0.0 && v == v.floor()`.
//!   `is_finite()` rejects Infinity/NaN so they fall through to `Float32`
//!   (original pyscx finding 9.4).
//! - Max-value comparison is done in `f64` so values beyond `2^24` (f32's
//!   contiguous integer range) don't saturate (finding 9.1).

use crate::dispatch::{CodecError, ValueEncoding};

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
    } else if max_val <= u32::MAX as f64 {
        ValueEncoding::Uint32
    } else {
        // f32 loses integer *contiguity* above 2²⁴ but still represents values
        // beyond u32::MAX exactly (2³², 1e10, …), so this bucket is reachable.
        // Encoding one as Uint32 would either saturate the `as u32` cast and
        // silently corrupt the value, or hard-fail `encode_f32`'s range check.
        ValueEncoding::Float32
    }
}

/// `f64` variant of [`detect_value_encoding`] for callers whose values
/// originate as `f64` (e.g. the R bindings' dgCMatrix `x` slot). Applies the
/// identical integer-detection policy and max-value buckets without a lossy
/// `f64 → f32` round-trip, so a value beyond `2^24` keeps full precision when
/// choosing between `Uint16`/`Uint32`.
pub fn detect_value_encoding_f64(data: &[f64]) -> ValueEncoding {
    let all_integer = data
        .iter()
        .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor());
    if !all_integer {
        return ValueEncoding::Float32;
    }
    let max_val: f64 = data.iter().copied().fold(0.0f64, f64::max);
    if max_val <= u8::MAX as f64 {
        ValueEncoding::Uint8
    } else if max_val <= u16::MAX as f64 {
        ValueEncoding::Uint16
    } else if max_val <= u32::MAX as f64 {
        ValueEncoding::Uint32
    } else {
        // f64 can exactly represent integers past u32::MAX (up to 2^53), so an
        // integer-valued f64 > u32::MAX must fall back to float — encoding it as
        // Uint32 would saturate the `f64 as u32` cast and silently corrupt the
        // value. The f32 detector has the identical bucket for the identical
        // reason: losing integer *contiguity* above 2^24 does not stop an f32
        // from representing 2^32 exactly.
        ValueEncoding::Float32
    }
}

/// Serialize f32 values to LE raw bytes according to `encoding`.
///
/// Thin wrapper over [`ValueEncoding::encode_f32_batch`] — kept as the
/// canonical entry point cited by historical call sites (`pyscx/anndata`,
/// `scx-cli/convert/dtype`, `scx-mtx/convert`). Returns
/// `Err(CodecError::Io(InvalidData))` if any value falls outside the
/// range representable by the requested integer encoding — with one
/// documented exception: under `Uint32`, exactly 2³² is accepted and
/// saturates to `u32::MAX`, because `u32::MAX as f32` *is* 2³² and the
/// encoder cannot tell a fresh out-of-range value from the f32 image of a
/// decoded `u32::MAX`. See [`ValueEncoding::encode_f32`].
pub fn values_to_raw_bytes(data: &[f32], encoding: ValueEncoding) -> Result<Vec<u8>, CodecError> {
    encoding.encode_f32_batch(data)
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
    fn detect_f64_buckets_match_f32() {
        assert_eq!(
            detect_value_encoding_f64(&[0.0, 1.0, 255.0]),
            ValueEncoding::Uint8
        );
        assert_eq!(
            detect_value_encoding_f64(&[256.0, 65535.0]),
            ValueEncoding::Uint16
        );
        assert_eq!(
            detect_value_encoding_f64(&[65536.0, 4_000_000_000.0]),
            ValueEncoding::Uint32
        );
        assert_eq!(
            detect_value_encoding_f64(&[1.5, 2.0]),
            ValueEncoding::Float32
        );
        assert_eq!(
            detect_value_encoding_f64(&[1.0, f64::NAN]),
            ValueEncoding::Float32
        );
        assert_eq!(
            detect_value_encoding_f64(&[1.0, -1.0]),
            ValueEncoding::Float32
        );
    }

    #[test]
    fn detect_f32_falls_back_to_float_above_u32_max() {
        // f32 loses integer *contiguity* above 2²⁴ but still represents 2³² and
        // 1e10 exactly, so the f32 detector reaches the same case as its f64
        // twin. `Uint32` would either saturate the `as u32` cast (2³² → u32::MAX,
        // silent corruption) or hard-fail the encoder's range check (1e10).
        let two_pow_32 = (1u128 << 32) as f32;
        assert_eq!(detect_value_encoding(&[two_pow_32]), ValueEncoding::Float32);
        assert_eq!(detect_value_encoding(&[1e10f32]), ValueEncoding::Float32);
        // The largest f32 at or below u32::MAX still picks Uint32.
        assert_eq!(
            detect_value_encoding(&[4_294_967_040.0f32]),
            ValueEncoding::Uint32
        );
    }

    #[test]
    fn detect_f64_falls_back_to_float_above_u32_max() {
        // 5e9 > u32::MAX (≈4.29e9) is an exact f64 integer; encoding it as Uint32
        // would saturate the `as u32` cast and corrupt the value, so it must fall
        // back to Float32. The f32 detector has the same bucket for the same
        // reason — see `detect_f32_falls_back_to_float_above_u32_max`.
        assert_eq!(
            detect_value_encoding_f64(&[5_000_000_000.0]),
            ValueEncoding::Float32
        );
        // Exactly u32::MAX still fits Uint32.
        assert_eq!(
            detect_value_encoding_f64(&[u32::MAX as f64]),
            ValueEncoding::Uint32
        );
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
        let bytes = values_to_raw_bytes(&data, ValueEncoding::Uint8).unwrap();
        let decoded: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn round_trip_uint16() {
        let data = [0.0_f32, 256.0, 65535.0];
        let bytes = values_to_raw_bytes(&data, ValueEncoding::Uint16).unwrap();
        let decoded: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn round_trip_float32() {
        let data = [0.0_f32, 1.5, -2.75, std::f32::consts::PI];
        let bytes = values_to_raw_bytes(&data, ValueEncoding::Float32).unwrap();
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn round_trip_float16_emits_2_byte_le() {
        let data = [0.0_f32, 1.5, -2.25, 100.0];
        let bytes = values_to_raw_bytes(&data, ValueEncoding::Float16).unwrap();
        assert_eq!(bytes.len(), data.len() * 2);
        let decoded: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert_eq!(half::f16::from_f32(*orig).to_f32(), *dec);
        }
    }

    #[test]
    fn float16_stride_differs_from_float32() {
        let data = [1.5_f32, 2.25];
        let f32_bytes = values_to_raw_bytes(&data, ValueEncoding::Float32).unwrap();
        let f16_bytes = values_to_raw_bytes(&data, ValueEncoding::Float16).unwrap();
        assert_eq!(f32_bytes.len(), data.len() * 4);
        assert_eq!(f16_bytes.len(), data.len() * 2);
    }
}
