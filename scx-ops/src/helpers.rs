// Shared helpers for scx-ops operations.

use scx_codec::ValueEncoding;

use crate::error::OpsError;

/// Encode a single f32 value back to raw bytes according to the value encoding.
///
/// Returns an error if the value is out of range for integer encodings.
/// Used by compact, merge, and other operations that re-encode decoded
/// float values back to their on-disk representation.
pub fn encode_value(
    buf: &mut Vec<u8>,
    value: f32,
    encoding: ValueEncoding,
) -> crate::error::Result<()> {
    match encoding {
        ValueEncoding::Uint8 => {
            if !(0.0..=u8::MAX as f32).contains(&value) {
                return Err(OpsError::ValueOutOfRange {
                    value,
                    encoding: "Uint8",
                    max: u8::MAX as f64,
                });
            }
            buf.push(value as u8);
        }
        ValueEncoding::Uint16 => {
            if !(0.0..=u16::MAX as f32).contains(&value) {
                return Err(OpsError::ValueOutOfRange {
                    value,
                    encoding: "Uint16",
                    max: u16::MAX as f64,
                });
            }
            buf.extend_from_slice(&(value as u16).to_le_bytes());
        }
        ValueEncoding::Uint32 => {
            // Exclusive at 2³², which f32 represents exactly. `u32::MAX as f32`
            // rounds *up* to 2³², so using it as an inclusive bound admits the
            // very value `as u32` saturates from and writes 4294967295 in its
            // place — silent corruption on the compact/merge re-encode path.
            // (255 and 65535 above are exact in f32, so those bounds are fine.)
            const UINT32_BOUND: f32 = (1u128 << 32) as f32;
            if !(0.0..UINT32_BOUND).contains(&value) {
                return Err(OpsError::ValueOutOfRange {
                    value,
                    encoding: "Uint32",
                    max: u32::MAX as f64,
                });
            }
            buf.extend_from_slice(&(value as u32).to_le_bytes());
        }
        ValueEncoding::Float32 => buf.extend_from_slice(&value.to_le_bytes()),
        ValueEncoding::Float16 => {
            let f16_val = half::f16::from_f32(value);
            buf.extend_from_slice(&f16_val.to_le_bytes());
        }
    }
    Ok(())
}

/// Pick one output value encoding wide enough to hold every input shard's
/// rows: any float source forces `Float32`, else the widest integer present.
///
/// Used wherever rows from multiple inputs (or shards) are re-packed into a
/// single output shard — merge X/layers (sorted and concat paths) — so the
/// chosen encoding can hold the widest value across all inputs rather than
/// truncating to the first shard's encoding.
pub(crate) fn widest_value_encoding(encs: &[ValueEncoding]) -> ValueEncoding {
    let mut any_float = false;
    let mut max_int = ValueEncoding::Uint8;
    for &e in encs {
        match e {
            ValueEncoding::Float32 | ValueEncoding::Float16 => any_float = true,
            ValueEncoding::Uint16 => {
                if matches!(max_int, ValueEncoding::Uint8) {
                    max_int = ValueEncoding::Uint16;
                }
            }
            ValueEncoding::Uint32 => max_int = ValueEncoding::Uint32,
            ValueEncoding::Uint8 => {}
        }
    }
    if any_float {
        ValueEncoding::Float32
    } else {
        max_int
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NaN is not a value any integer encoding can hold. The range checks are
    /// written as `!range.contains(&v)` rather than a pair of comparisons
    /// precisely because `contains` is false for NaN, so all three arms reject
    /// it instead of writing `value as uN` (which is 0).
    #[test]
    fn encode_value_integer_encodings_reject_nan() {
        for enc in [
            ValueEncoding::Uint8,
            ValueEncoding::Uint16,
            ValueEncoding::Uint32,
        ] {
            let mut buf = Vec::new();
            assert!(
                matches!(
                    encode_value(&mut buf, f32::NAN, enc),
                    Err(OpsError::ValueOutOfRange { .. })
                ),
                "NaN accepted by {enc:?}"
            );
            assert!(buf.is_empty());
        }
    }

    /// The re-encode path used by compact/merge shares the codec's saturation
    /// hazard: `u32::MAX as f32` rounds up to 2³², so an inclusive bound
    /// written that way lets 2³² through and `as u32` saturates it to
    /// 4294967295 on the way to disk.
    #[test]
    fn encode_value_uint32_rejects_saturating_value() {
        let two_pow_32 = (1u128 << 32) as f32;
        let mut buf = Vec::new();
        assert!(matches!(
            encode_value(&mut buf, two_pow_32, ValueEncoding::Uint32),
            Err(OpsError::ValueOutOfRange { .. })
        ));
        assert!(buf.is_empty());
        // The largest f32 below the bound (2³² - 2⁸) still encodes, exactly.
        encode_value(&mut buf, 4_294_967_040.0, ValueEncoding::Uint32).unwrap();
        assert_eq!(buf, 4_294_967_040u32.to_le_bytes());
    }
}
