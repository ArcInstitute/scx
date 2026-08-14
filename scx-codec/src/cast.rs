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

/// Signed-integer value target: an explicit range guard, then a round-trip
/// equality check for fractional loss.
///
/// The range guard cannot be written as `<$t>::MAX as f32`. Rust's `f32 as int`
/// *saturates*, and for a target with ≥24 significant bits `<$t>::MAX as f32`
/// rounds **up** to `2^(B-1)` — precisely the value the cast saturates from —
/// so the two alias and a round-trip check performed in `f32` cannot tell a
/// saturated result from an exact one (`i32`, `i64`). Bound against the true
/// exclusive limit instead: `[-2^(B-1), 2^(B-1))`, both powers of two and
/// therefore exact in `f32` at every width.
macro_rules! impl_cast_from_f32_signed_int {
    ($t:ty, $name:literal) => {
        impl CastFromF32 for $t {
            const NAME: &'static str = $name;
            const ALWAYS_SAFE: bool = false;
            fn cast_unchecked(v: f32) -> Self {
                v as $t
            }
            fn cast_lossless(v: f32) -> Option<Self> {
                // `const`, not `let`: a `u128 as f32` conversion is a
                // compiler-rt call that an unoptimized build would otherwise
                // emit per element in this hot loop.
                const BOUND: f32 = (1u128 << (<$t>::BITS - 1)) as f32;
                if !v.is_finite() || !(-BOUND..BOUND).contains(&v) {
                    return None;
                }
                // In range, so the cast below cannot saturate: the round-trip
                // is purely a fractional-loss check.
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

/// Unsigned-integer value target: a sign-loss guard (negative → unsigned) and
/// the same exact-in-`f32` exclusive upper bound as the signed macro (`2^B`
/// here), then the round-trip fractional check.
macro_rules! impl_cast_from_f32_unsigned_int {
    ($t:ty, $name:literal) => {
        impl CastFromF32 for $t {
            const NAME: &'static str = $name;
            const ALWAYS_SAFE: bool = false;
            fn cast_unchecked(v: f32) -> Self {
                v as $t
            }
            fn cast_lossless(v: f32) -> Option<Self> {
                const BOUND: f32 = (1u128 << <$t>::BITS) as f32;
                if !v.is_finite() || !(0.0..BOUND).contains(&v) {
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

// --- Native integer source (u32) ------------------------------------------
//
// The in-assembly narrow read decodes an integer-encoded shard to its native
// `u32` stream (Scx1 `rice_decode`, or raw-LE `u8`/`u16`/`u32` widened to
// `u32`) and casts *directly* to the target dtype, never routing through `f32`.
// This makes integer→integer narrows exact for any `value_max` (including
// counts `> 2²⁴`, which the f32 path rounds) — that is the correctness win the
// `CastFromF32` path cannot deliver.

/// A target dtype reachable from a `u32` value stream.
pub trait CastFromU32: Copy + Sized {
    /// numpy dtype name, used in error messages.
    const NAME: &'static str;
    /// `true` when every `u32` casts without loss (`u32` ⊆ target exactly).
    const ALWAYS_SAFE: bool;
    /// Largest `u32` that casts to this target losslessly. Drives the O(1)
    /// pre-decode guard: integer range limits (`u16` → 65535) and float
    /// exact-integer limits (`f16` → 2048, `f32` → 2²⁴) both fall out of it.
    const MAX_LOSSLESS_U32: u32;
    /// Plain cast, no checking. Only called when known safe or `allow_lossy`.
    fn cast_unchecked(v: u32) -> Self;
    /// `Some(cast)` iff the value casts losslessly, else `None`.
    fn cast_lossless(v: u32) -> Option<Self>;
}

/// Integer value target from `u32`: `TryFrom<u32>` is the exact range check
/// (exact for any `value_max`, including `> 2²⁴`, because `f32` is never
/// involved).
macro_rules! impl_cast_from_u32_int {
    ($t:ty, $name:literal, $always:expr, $max_lossless:expr) => {
        impl CastFromU32 for $t {
            const NAME: &'static str = $name;
            const ALWAYS_SAFE: bool = $always;
            const MAX_LOSSLESS_U32: u32 = $max_lossless;
            fn cast_unchecked(v: u32) -> Self {
                v as $t
            }
            fn cast_lossless(v: u32) -> Option<Self> {
                <$t>::try_from(v).ok()
            }
        }
    };
}

impl_cast_from_u32_int!(u8, "uint8", false, 255);
impl_cast_from_u32_int!(u16, "uint16", false, 65535);
impl_cast_from_u32_int!(u32, "uint32", true, u32::MAX);
impl_cast_from_u32_int!(i8, "int8", false, 127);
impl_cast_from_u32_int!(i16, "int16", false, 32767);
impl_cast_from_u32_int!(i32, "int32", false, i32::MAX as u32);
impl_cast_from_u32_int!(i64, "int64", true, u32::MAX);

impl CastFromU32 for f64 {
    const NAME: &'static str = "float64";
    const ALWAYS_SAFE: bool = true;
    const MAX_LOSSLESS_U32: u32 = u32::MAX;
    fn cast_unchecked(v: u32) -> Self {
        v as f64
    }
    fn cast_lossless(v: u32) -> Option<Self> {
        Some(v as f64)
    }
}

impl CastFromU32 for f32 {
    const NAME: &'static str = "float32";
    const ALWAYS_SAFE: bool = false;
    const MAX_LOSSLESS_U32: u32 = F32_MAX_EXACT_INT;
    fn cast_unchecked(v: u32) -> Self {
        v as f32
    }
    fn cast_lossless(v: u32) -> Option<Self> {
        let t = v as f32;
        // Exact iff the f32 round-trips to the same integer. Compare in f64,
        // which represents every u32 exactly, so the check itself is exact.
        if (t as f64) == v as f64 {
            Some(t)
        } else {
            None
        }
    }
}

impl CastFromU32 for half::f16 {
    const NAME: &'static str = "float16";
    const ALWAYS_SAFE: bool = false;
    const MAX_LOSSLESS_U32: u32 = 2048;
    fn cast_unchecked(v: u32) -> Self {
        half::f16::from_f32(v as f32)
    }
    fn cast_lossless(v: u32) -> Option<Self> {
        let t = half::f16::from_f32(v as f32);
        if (t.to_f32() as f64) == v as f64 {
            Some(t)
        } else {
            None
        }
    }
}

/// Cast a `u32` value stream to `T`, failing loud on any lossy narrowing unless
/// `allow_lossy` is set. Mirror of [`checked_cast_values`] with a `u32` source.
pub fn checked_cast_values_u32<T: CastFromU32>(
    src: &[u32],
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

/// Length mismatch between an in-place cast's source and destination slices.
fn cast_into_len_check(src_len: usize, dst_len: usize) -> Result<(), CodecError> {
    if src_len != dst_len {
        return Err(CodecError::MalformedInput(format!(
            "cast_into length mismatch: {src_len} source values into {dst_len} destination slots"
        )));
    }
    Ok(())
}

/// Cast a `u32` stream into a caller-owned destination slice (the in-assembly
/// reader fills a shard's slice of the target buffer at its running nnz offset).
pub fn checked_cast_u32_into<T: CastFromU32>(
    src: &[u32],
    dst: &mut [T],
    allow_lossy: bool,
) -> Result<(), CodecError> {
    cast_into_len_check(src.len(), dst.len())?;
    if T::ALWAYS_SAFE || allow_lossy {
        for (d, &v) in dst.iter_mut().zip(src) {
            *d = T::cast_unchecked(v);
        }
        return Ok(());
    }
    for (i, (d, &v)) in dst.iter_mut().zip(src).enumerate() {
        match T::cast_lossless(v) {
            Some(t) => *d = t,
            None => {
                return Err(CodecError::MalformedInput(format!(
                    "lossy cast of value {v} (at nnz index {i}) to {name}; \
                     use a wider `data_dtype` or pass `allow_lossy=True`",
                    name = T::NAME,
                )))
            }
        }
    }
    Ok(())
}

/// Cast an `f32` stream into a caller-owned destination slice (float-encoded
/// shards decode native → `f32`, then narrow via this in-place variant).
pub fn checked_cast_f32_into<T: CastFromF32>(
    src: &[f32],
    dst: &mut [T],
    allow_lossy: bool,
) -> Result<(), CodecError> {
    cast_into_len_check(src.len(), dst.len())?;
    if T::ALWAYS_SAFE || allow_lossy {
        for (d, &v) in dst.iter_mut().zip(src) {
            *d = T::cast_unchecked(v);
        }
        return Ok(());
    }
    for (i, (d, &v)) in dst.iter_mut().zip(src).enumerate() {
        match T::cast_lossless(v) {
            Some(t) => *d = t,
            None => {
                return Err(CodecError::MalformedInput(format!(
                    "lossy cast of value {v} (at nnz index {i}) to {name}; \
                     use a wider `data_dtype` or pass `allow_lossy=True`",
                    name = T::NAME,
                )))
            }
        }
    }
    Ok(())
}

/// Cast an `i32` index stream into a caller-owned destination slice.
pub fn checked_cast_i32_into<I: CastFromI32>(
    src: &[i32],
    dst: &mut [I],
    allow_lossy: bool,
) -> Result<(), CodecError> {
    cast_into_len_check(src.len(), dst.len())?;
    if I::ALWAYS_SAFE || allow_lossy {
        for (d, &v) in dst.iter_mut().zip(src) {
            *d = I::cast_unchecked(v);
        }
        return Ok(());
    }
    for (i, (d, &v)) in dst.iter_mut().zip(src).enumerate() {
        match I::cast_lossless(v) {
            Some(t) => *d = t,
            None => {
                return Err(CodecError::MalformedInput(format!(
                    "lossy cast of column index {v} (at nnz index {i}) to {name}; \
                     use a wider `index_dtype`",
                    name = I::NAME,
                )))
            }
        }
    }
    Ok(())
}

/// O(1) pre-decode guard for a chosen target dtype: fails loud when an
/// integer-encoded shard's `max_value` cannot be represented losslessly in the
/// target `T` and the caller has not opted into lossy narrowing.
///
/// The dtype-aware generalization of [`guard_f32_decode_loss`]: keyed on
/// `T::MAX_LOSSLESS_U32`, so it covers float exact-integer limits (`f16` → 2048,
/// `f32` → 2²⁴, `f64` → never) and integer range limits (`u16` → 65535, …)
/// uniformly. Float-encoded shards record `value_max = 0`, so they never trip
/// it regardless of target.
pub fn guard_decode_loss_for<T: CastFromU32>(
    max_value: u32,
    allow_lossy: bool,
) -> Result<(), CodecError> {
    if allow_lossy || max_value <= T::MAX_LOSSLESS_U32 {
        return Ok(());
    }
    Err(CodecError::MalformedInput(format!(
        "integer value {max_value} exceeds {limit}, the largest integer representable \
         exactly in {name}; this read would lose precision. Use a wider `data_dtype` or \
         pass `allow_lossy=True`.",
        limit = T::MAX_LOSSLESS_U32,
        name = T::NAME,
    )))
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
    fn f32_saturation_alias_rejected() {
        // Rust's `f32 as int` saturates, and for a target with >= 24
        // significant bits `T::MAX as f32` rounds *up* to `T::MAX + 1` — so a
        // round-trip check performed in f32 cannot tell a saturated result from
        // an exact one. Each of these must fail loud, not return `T::MAX`.
        let two_pow_31 = (1u128 << 31) as f32;
        let two_pow_32 = (1u128 << 32) as f32;
        let two_pow_63 = (1u128 << 63) as f32;
        assert!(checked_cast_values::<i32>(&[two_pow_31], false).is_err());
        assert!(checked_cast_values::<u32>(&[two_pow_32], false).is_err());
        assert!(checked_cast_values::<i64>(&[two_pow_63], false).is_err());
        // Narrow targets were never affected; keep them covered so a future
        // rewrite cannot regress them.
        assert!(checked_cast_values::<i8>(&[128.0f32], false).is_err());
        assert!(checked_cast_values::<i16>(&[32768.0f32], false).is_err());
        assert!(checked_cast_values::<u8>(&[256.0f32], false).is_err());
        assert!(checked_cast_values::<u16>(&[65536.0f32], false).is_err());
        // Past the negative bound (-2^31 - 2^8, the f32 below i32::MIN).
        assert!(checked_cast_values::<i32>(&[-2147483904.0f32], false).is_err());
        // `allow_lossy` still saturates without error (documented escape hatch).
        let out: Vec<i32> = checked_cast_values(&[two_pow_31], true).unwrap();
        assert_eq!(out, vec![i32::MAX]);
    }

    #[test]
    fn f32_largest_representable_still_accepted() {
        // The saturation guard must not over-reject: the largest f32 strictly
        // below each target's exclusive bound is exact and in range, and each
        // MIN is a power of two so the lower bound is inclusive.
        assert_eq!(
            checked_cast_values::<i8>(&[127.0, -128.0], false).unwrap(),
            vec![127i8, -128]
        );
        assert_eq!(
            checked_cast_values::<i16>(&[32767.0, -32768.0], false).unwrap(),
            vec![32767i16, -32768]
        );
        assert_eq!(
            checked_cast_values::<u8>(&[255.0], false).unwrap(),
            vec![255u8]
        );
        assert_eq!(
            checked_cast_values::<u16>(&[65535.0], false).unwrap(),
            vec![65535u16]
        );
        // 2^31 - 2^7 and 2^32 - 2^8: the f32s immediately below the bounds.
        assert_eq!(
            checked_cast_values::<i32>(&[2147483520.0f32], false).unwrap(),
            vec![2147483520i32]
        );
        assert_eq!(
            checked_cast_values::<u32>(&[4294967040.0f32], false).unwrap(),
            vec![4294967040u32]
        );
        // 2^63 - 2^39 (f32 spacing at that magnitude is 2^39).
        let i64_top = ((1u128 << 63) - (1u128 << 39)) as f32;
        assert_eq!(
            checked_cast_values::<i64>(&[i64_top], false).unwrap(),
            vec![9223371487098961920i64]
        );
        // i64::MIN is exactly -2^63 and the lower bound is inclusive.
        let i64_min = -((1u128 << 63) as f32);
        assert_eq!(
            checked_cast_values::<i64>(&[i64_min], false).unwrap(),
            vec![i64::MIN]
        );
    }

    #[test]
    fn f32_into_rejects_saturation_alias() {
        // The in-assembly narrow reader reaches these impls through
        // `checked_cast_f32_into`, not `checked_cast_values`.
        let mut dst = [0i32; 1];
        assert!(checked_cast_f32_into(&[(1u128 << 31) as f32], &mut dst, false).is_err());
        let mut ok = [0i32; 1];
        checked_cast_f32_into(&[2147483520.0f32], &mut ok, false).unwrap();
        assert_eq!(ok, [2147483520i32]);
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

    // --- Native u32-source gate ---

    #[test]
    fn u32_source_exact_above_2pow24() {
        // The whole point: integer counts > 2²⁴ narrow losslessly to exact
        // integer targets, where the f32 path would round.
        let vals = [
            F32_MAX_EXACT_INT - 1,
            F32_MAX_EXACT_INT,
            F32_MAX_EXACT_INT + 1,
            u32::MAX,
        ];
        for &v in &vals {
            // u32 target (identity, ALWAYS_SAFE) and i64 target are exact.
            let u: Vec<u32> = checked_cast_values_u32(&[v], false).unwrap();
            assert_eq!(u, vec![v]);
            let i: Vec<i64> = checked_cast_values_u32(&[v], false).unwrap();
            assert_eq!(i, vec![v as i64]);
            // u16 cannot represent any of these (all > 65535) -> fail loud.
            assert!(checked_cast_values_u32::<u16>(&[v], false).is_err());
        }
    }

    #[test]
    fn u32_narrow_out_of_range_fails_loud() {
        let err = checked_cast_values_u32::<u8>(&[256], false).unwrap_err();
        assert!(matches!(err, CodecError::MalformedInput(_)));
        // allow_lossy saturates/truncates without error (256 as u8 == 0).
        let out: Vec<u8> = checked_cast_values_u32(&[256], true).unwrap();
        assert_eq!(out, vec![0u8]);
    }

    #[test]
    fn u32_into_fills_slice() {
        let src = [1u32, 2, 3];
        let mut dst = [0u16; 3];
        checked_cast_u32_into(&src, &mut dst, false).unwrap();
        assert_eq!(dst, [1u16, 2, 3]);
        // length mismatch is an error
        let mut short = [0u16; 2];
        assert!(checked_cast_u32_into(&src, &mut short, false).is_err());
        // lossy element without allow_lossy errors, leaving the destination untouched past the fault
        let mut d2 = [0u8; 2];
        assert!(checked_cast_u32_into(&[10u32, 300], &mut d2, false).is_err());
    }

    #[test]
    fn f32_and_i32_into_fill_slices() {
        let mut vd = [0.0f64; 3];
        checked_cast_f32_into(&[1.0f32, 2.5, -3.0], &mut vd, false).unwrap();
        assert_eq!(vd, [1.0f64, 2.5, -3.0]);

        let mut idx = [0i16; 3];
        checked_cast_i32_into(&[0i32, 5, 32767], &mut idx, false).unwrap();
        assert_eq!(idx, [0i16, 5, 32767]);
        let mut idx_bad = [0i16; 1];
        assert!(checked_cast_i32_into(&[32768i32], &mut idx_bad, false).is_err());
    }

    #[test]
    fn guard_decode_loss_for_thresholds() {
        // Integer targets: fire on range overflow.
        assert!(guard_decode_loss_for::<u16>(65535, false).is_ok());
        assert!(guard_decode_loss_for::<u16>(65536, false).is_err());
        // Always-safe targets never fire.
        assert!(guard_decode_loss_for::<u32>(u32::MAX, false).is_ok());
        assert!(guard_decode_loss_for::<i64>(u32::MAX, false).is_ok());
        assert!(guard_decode_loss_for::<f64>(u32::MAX, false).is_ok());
        // Float exact-integer limits.
        assert!(guard_decode_loss_for::<half::f16>(2048, false).is_ok());
        assert!(guard_decode_loss_for::<half::f16>(2049, false).is_err());
        assert!(guard_decode_loss_for::<f32>(F32_MAX_EXACT_INT, false).is_ok());
        assert!(guard_decode_loss_for::<f32>(F32_MAX_EXACT_INT + 1, false).is_err());
        // allow_lossy bypasses regardless of target.
        assert!(guard_decode_loss_for::<u16>(65536, true).is_ok());
        assert!(guard_decode_loss_for::<f32>(u32::MAX, true).is_ok());
    }

    #[test]
    fn guard_f32_parity_with_generic() {
        // The f32 specialization must agree with the generic guard at the boundary.
        for &v in &[F32_MAX_EXACT_INT, F32_MAX_EXACT_INT + 1, u32::MAX] {
            assert_eq!(
                guard_f32_decode_loss(v, false).is_ok(),
                guard_decode_loss_for::<f32>(v, false).is_ok(),
                "mismatch at {v}"
            );
        }
    }
}
