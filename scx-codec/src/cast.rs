// Fail-loud lossless-cast gate for read-side numeric narrowing.
//
// A single gate before any narrowing cast: a sign-loss guard (negative →
// unsigned) plus a round-trip equality check. Widening casts (f32→f64, i32→i64,
// identity) are recognized as always-safe and skip the per-element check.
//
// Source types are fixed by scx's read path: values are `f32` (from the *already
// decoded* CSR), indices are `i32`. Targets are the numpy-representable numeric
// dtypes. On a lossy cast without `allow_lossy`, returns
// `CodecError::MalformedInput`.
//
// Scope: `checked_cast_*` gates narrowing *from the decoded f32 stream* only.
// The decode-level `u32 as f32` casts (dispatch.rs, e.g. the Scx1 rice fast path
// and `values_raw_to_f32`) are upstream and still round integer counts above 2²⁴
// to f32 before that gate ever sees them. `guard_f32_decode_loss` (below) is the
// O(1) companion that a reader calls *before* decode, using the catalog's
// `value_max`, so a read of such an archive fails loud instead of silently
// rounding. Making a >2²⁴ integer read *succeed* (typed lossless decode) is
// still Phase 2 (push-dtype-into-decode), not this module.

use crate::dispatch::CodecError;

/// Largest integer that `f32` represents exactly (2²⁴). Above this, consecutive
/// integers are no longer all representable, so decoding `u32` counts to `f32`
/// rounds.
pub const F32_MAX_EXACT_INT: u32 = 1 << 24;

/// Fail loud when an integer shard's `value_max` exceeds what `f32` represents
/// exactly and the caller has not opted into lossy narrowing.
///
/// scx's Phase-1 read decodes every shard to an `f32` CSR before any dtype
/// materialization, so an on-disk `u32` count above 2²⁴ is silently rounded
/// regardless of the requested output dtype. This O(1) check runs *before*
/// decode using the catalog's `value_max` (integer-encoded shards only —
/// float-encoded shards record `value_max = 0`, so they never trip it).
///
/// `max_value` is the maximum `ShardStats::value_max` over the shards in scope
/// of the read. Returns `CodecError::MalformedInput` when the decode would lose
/// precision and `allow_lossy` is `false`.
pub fn guard_f32_decode_loss(max_value: u32, allow_lossy: bool) -> Result<(), CodecError> {
    if allow_lossy || max_value <= F32_MAX_EXACT_INT {
        return Ok(());
    }
    Err(CodecError::MalformedInput(format!(
        "integer value {max_value} exceeds 2²⁴ ({F32_MAX_EXACT_INT}), the largest integer \
         representable exactly in float32; scx decodes counts to f32 before materialization, \
         so this read would silently round large counts. Pass `allow_lossy=True` to accept \
         the f32 rounding (lossless typed decode is not yet available)."
    )))
}

/// A target dtype reachable from an `f32` value stream.
pub trait CastFromF32: Copy + Sized {
    /// numpy dtype name, used in error messages.
    const NAME: &'static str;
    /// `true` when every `f32` casts without loss (widening / identity).
    const ALWAYS_SAFE: bool;
    /// Plain cast, no checking. Only called when the cast is known safe or the
    /// caller opted into `allow_lossy`.
    fn cast_unchecked(v: f32) -> Self;
    /// Returns `Some(cast)` iff the value casts losslessly (round-trips exactly
    /// and does not lose sign), else `None`.
    fn cast_lossless(v: f32) -> Option<Self>;
}

/// A target dtype reachable from an `i32` index stream.
pub trait CastFromI32: Copy + Sized {
    const NAME: &'static str;
    const ALWAYS_SAFE: bool;
    fn cast_unchecked(v: i32) -> Self;
    fn cast_lossless(v: i32) -> Option<Self>;
}

/// Cast an `f32` value stream to `T`, failing loud on any lossy narrowing
/// unless `allow_lossy` is set.
pub fn checked_cast_values<T: CastFromF32>(
    src: &[f32],
    allow_lossy: bool,
) -> Result<Vec<T>, CodecError> {
    if T::ALWAYS_SAFE || allow_lossy {
        return Ok(src.iter().map(|&v| T::cast_unchecked(v)).collect());
    }
    let mut out = Vec::with_capacity(src.len());
    for (i, &v) in src.iter().enumerate() {
        match T::cast_lossless(v) {
            Some(t) => out.push(t),
            None => {
                return Err(CodecError::MalformedInput(format!(
                    "lossy cast of value {v} (at nnz index {i}) to {name}; \
                     use a wider `data_dtype` or pass `allow_lossy=True`",
                    name = T::NAME,
                )))
            }
        }
    }
    Ok(out)
}

/// Cast an `i32` index stream to `I`, failing loud on any lossy narrowing
/// unless `allow_lossy` is set.
pub fn checked_cast_indices<I: CastFromI32>(
    src: &[i32],
    allow_lossy: bool,
) -> Result<Vec<I>, CodecError> {
    if I::ALWAYS_SAFE || allow_lossy {
        return Ok(src.iter().map(|&v| I::cast_unchecked(v)).collect());
    }
    let mut out = Vec::with_capacity(src.len());
    for (i, &v) in src.iter().enumerate() {
        match I::cast_lossless(v) {
            Some(t) => out.push(t),
            None => {
                return Err(CodecError::MalformedInput(format!(
                    "lossy cast of column index {v} (at nnz index {i}) to {name}; \
                     use a wider `index_dtype`",
                    name = I::NAME,
                )))
            }
        }
    }
    Ok(out)
}

// --- Value target impls (source: f32) ---

impl CastFromF32 for f32 {
    const NAME: &'static str = "float32";
    const ALWAYS_SAFE: bool = true;
    fn cast_unchecked(v: f32) -> Self {
        v
    }
    fn cast_lossless(v: f32) -> Option<Self> {
        Some(v)
    }
}

impl CastFromF32 for f64 {
    const NAME: &'static str = "float64";
    const ALWAYS_SAFE: bool = true;
    fn cast_unchecked(v: f32) -> Self {
        v as f64
    }
    fn cast_lossless(v: f32) -> Option<Self> {
        Some(v as f64)
    }
}

impl CastFromF32 for half::f16 {
    const NAME: &'static str = "float16";
    const ALWAYS_SAFE: bool = false;
    fn cast_unchecked(v: f32) -> Self {
        half::f16::from_f32(v)
    }
    fn cast_lossless(v: f32) -> Option<Self> {
        let t = half::f16::from_f32(v);
        // NaN is special-cased because the round-trip check below is
        // equality-based and `NaN != NaN` — an f32 NaN would always be flagged
        // lossy otherwise. f16 represents NaN, so the cast is not lossy: accept
        // any NaN→NaN. (Infinity round-trips through the equality check fine.)
        if v.is_nan() {
            if t.is_nan() {
                return Some(t);
            }
            return None;
        }
        if t.to_f32() == v {
            Some(t)
        } else {
            None
        }
    }
}

/// Signed-integer value target: round-trip through `f32` catches range and
/// fractional loss (Rust's `f32 as int` saturates, so out-of-range fails the
/// equality check).
macro_rules! impl_cast_from_f32_signed_int {
    ($t:ty, $name:literal) => {
        impl CastFromF32 for $t {
            const NAME: &'static str = $name;
            const ALWAYS_SAFE: bool = false;
            fn cast_unchecked(v: f32) -> Self {
                v as $t
            }
            fn cast_lossless(v: f32) -> Option<Self> {
                if !v.is_finite() {
                    return None;
                }
                let t = v as $t;
                if t as f32 == v {
                    Some(t)
                } else {
                    None
                }
            }
        }
    };
}

/// Unsigned-integer value target: explicit sign-loss guard (negative → unsigned)
/// plus the round-trip equality check.
macro_rules! impl_cast_from_f32_unsigned_int {
    ($t:ty, $name:literal) => {
        impl CastFromF32 for $t {
            const NAME: &'static str = $name;
            const ALWAYS_SAFE: bool = false;
            fn cast_unchecked(v: f32) -> Self {
                v as $t
            }
            fn cast_lossless(v: f32) -> Option<Self> {
                if !v.is_finite() || v < 0.0 {
                    return None;
                }
                let t = v as $t;
                if t as f32 == v {
                    Some(t)
                } else {
                    None
                }
            }
        }
    };
}

impl_cast_from_f32_signed_int!(i8, "int8");
impl_cast_from_f32_signed_int!(i16, "int16");
impl_cast_from_f32_signed_int!(i32, "int32");
impl_cast_from_f32_signed_int!(i64, "int64");
impl_cast_from_f32_unsigned_int!(u8, "uint8");
impl_cast_from_f32_unsigned_int!(u16, "uint16");
impl_cast_from_f32_unsigned_int!(u32, "uint32");

// --- Index target impls (source: i32) ---

impl CastFromI32 for i32 {
    const NAME: &'static str = "int32";
    const ALWAYS_SAFE: bool = true;
    fn cast_unchecked(v: i32) -> Self {
        v
    }
    fn cast_lossless(v: i32) -> Option<Self> {
        Some(v)
    }
}

impl CastFromI32 for i64 {
    const NAME: &'static str = "int64";
    const ALWAYS_SAFE: bool = true;
    fn cast_unchecked(v: i32) -> Self {
        v as i64
    }
    fn cast_lossless(v: i32) -> Option<Self> {
        Some(v as i64)
    }
}

impl CastFromI32 for i16 {
    const NAME: &'static str = "int16";
    const ALWAYS_SAFE: bool = false;
    fn cast_unchecked(v: i32) -> Self {
        v as i16
    }
    fn cast_lossless(v: i32) -> Option<Self> {
        let t = v as i16;
        if t as i32 == v {
            Some(t)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widen_is_always_safe_scanfree() {
        // f32 -> f64 / f32 identity: no error even without allow_lossy.
        let v = vec![1.0f32, 2.5, -3.0];
        let out: Vec<f64> = checked_cast_values(&v, false).unwrap();
        assert_eq!(out, vec![1.0f64, 2.5, -3.0]);
        let out32: Vec<f32> = checked_cast_values(&v, false).unwrap();
        assert_eq!(out32, v);
    }

    #[test]
    fn integer_narrow_lossless_ok() {
        let v = vec![0.0f32, 255.0, 42.0];
        let out: Vec<u8> = checked_cast_values(&v, false).unwrap();
        assert_eq!(out, vec![0u8, 255, 42]);
    }

    #[test]
    fn integer_narrow_out_of_range_fails_loud() {
        let v = vec![0.0f32, 256.0];
        let err = checked_cast_values::<u8>(&v, false).unwrap_err();
        assert!(matches!(err, CodecError::MalformedInput(_)));
        // with allow_lossy it truncates (saturates) without error
        let out: Vec<u8> = checked_cast_values(&v, true).unwrap();
        assert_eq!(out, vec![0u8, 255]);
    }

    #[test]
    fn sign_loss_into_unsigned_fails_loud() {
        // int8(-1) -> uint16 analog: f32(-1.0) -> u16 must raise.
        let v = vec![-1.0f32];
        let err = checked_cast_values::<u16>(&v, false).unwrap_err();
        assert!(matches!(err, CodecError::MalformedInput(_)));
    }

    #[test]
    fn fractional_into_int_fails_loud() {
        let v = vec![1.5f32];
        assert!(checked_cast_values::<i32>(&v, false).is_err());
    }

    #[test]
    fn u32_above_2pow24_into_f16_fails_loud() {
        let v = vec![20_000_000.0f32];
        let err = checked_cast_values::<half::f16>(&v, false).unwrap_err();
        assert!(matches!(err, CodecError::MalformedInput(_)));
    }

    #[test]
    fn f16_representable_ok() {
        let v = vec![0.0f32, 1.0, 0.5, -2.0, 100.0];
        let out: Vec<half::f16> = checked_cast_values(&v, false).unwrap();
        let back: Vec<f32> = out.iter().map(|h| h.to_f32()).collect();
        assert_eq!(back, v);
    }

    #[test]
    fn guard_f32_decode_loss_thresholds() {
        // Below and at the exact-int limit: never errors.
        assert!(guard_f32_decode_loss(0, false).is_ok());
        assert!(guard_f32_decode_loss(255, false).is_ok());
        assert!(guard_f32_decode_loss(F32_MAX_EXACT_INT, false).is_ok());
        // Above the limit without allow_lossy: fail loud.
        let err = guard_f32_decode_loss(F32_MAX_EXACT_INT + 1, false).unwrap_err();
        assert!(matches!(err, CodecError::MalformedInput(_)));
        assert!(guard_f32_decode_loss(u32::MAX, false).is_err());
        // Above the limit with allow_lossy: OK (caller accepts rounding).
        assert!(guard_f32_decode_loss(F32_MAX_EXACT_INT + 1, true).is_ok());
        assert!(guard_f32_decode_loss(u32::MAX, true).is_ok());
    }

    #[test]
    fn index_widen_and_narrow() {
        let idx = vec![0i32, 32767, 5];
        let i64s: Vec<i64> = checked_cast_indices(&idx, false).unwrap();
        assert_eq!(i64s, vec![0i64, 32767, 5]);
        let i16s: Vec<i16> = checked_cast_indices(&idx, false).unwrap();
        assert_eq!(i16s, vec![0i16, 32767, 5]);
        // 32768 overflows i16 -> fail loud
        let big = vec![32768i32];
        assert!(checked_cast_indices::<i16>(&big, false).is_err());
    }
}
