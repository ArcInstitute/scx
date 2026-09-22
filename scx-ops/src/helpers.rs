// Shared helpers for scx-ops operations.

use scx_codec::ValueEncoding;

use crate::error::OpsError;

/// Encode a single f32 value back to raw bytes according to the value encoding.
///
/// Returns an error if the value is out of range for integer encodings.
/// Used by compact, merge, and other operations that re-encode decoded
/// float values back to their on-disk representation.
///
/// Thin over [`ValueEncoding::encode_f32`], which owns the range rules — the
/// inclusive-at-2³² `Uint32` bound that lets a decoded `u32::MAX` round-trip,
/// and the `contains` test that rejects NaN instead of writing `value as uN` as
/// a silent `0`. This crate previously restated all three arms, so the bounds
/// were declared twice and could drift; what it actually needs is not a second
/// range check but its **own error type**, since
/// [`OpsError::ValueOutOfRange`]'s `value` / `encoding` / `max` fields are what
/// pyscx maps to a Python exception and what the tests below assert on.
pub fn encode_value(
    buf: &mut Vec<u8>,
    value: f32,
    encoding: ValueEncoding,
) -> crate::error::Result<()> {
    encoding.encode_f32(buf, value).map_err(lift_range_error)
}

/// [`encode_value`] over a whole contiguous run of values.
///
/// The rewrite ops' accumulator loops call this once per kept row rather than
/// once per nonzero. Measured on `scx compact --codec auto` at census_1m,
/// `encode_value` was **5.08 % of all cycles**, and because it runs on the
/// calling thread while the encode runs on the pool, that is roughly **16 % of
/// the op's wall** — the largest serial term left after PR-42.
///
/// The width-specialised work is [`ValueEncoding::encode_f32_into`], so there is
/// one implementation in the workspace — 1.7-2.1x on the integer widths and
/// 27.8x on `Float32`, benched there. That is short of the 5-10x the OPT plan
/// projected for this pass, so expect roughly 8 % off a census_1m compact from
/// it, not 14 %.
///
/// `buf` is left exactly as it was found on error: an accumulator must not gain
/// half a row behind an `Err`, because the failed row wrote nothing.
pub fn encode_values(
    buf: &mut Vec<u8>,
    data: &[f32],
    encoding: ValueEncoding,
) -> crate::error::Result<()> {
    encoding
        .encode_f32_into(buf, data)
        .map_err(lift_range_error)
}

/// Lift `scx-codec`'s typed range error into this crate's, preserving the
/// offending value.
///
/// Only that one variant is re-shaped; everything else keeps its existing
/// classification through `OpsError::Codec`. Before `CodecError` carried the
/// value, this crate had to re-run the per-value loop after a batch failure
/// just to rediscover which value was bad — two range passes and two error
/// paths for one failure.
fn lift_range_error(e: scx_codec::CodecError) -> OpsError {
    match e {
        scx_codec::CodecError::ValueOutOfRange {
            value,
            encoding,
            max,
        } => OpsError::ValueOutOfRange {
            value,
            encoding,
            max,
        },
        other => OpsError::from(other),
    }
}

/// Pick one output value encoding wide enough to hold every input shard's
/// rows: any float source forces `Float32`, else the widest integer present.
///
/// Used wherever rows from multiple inputs (or shards) are re-packed into a
/// single output shard — merge X/layers (sorted and concat paths) — so the
/// chosen encoding can hold the widest value across all inputs rather than
/// truncating to the first shard's encoding.
///
/// A one-line delegate: the rule itself lives in `scx-codec` as
/// [`ValueEncoding::widest_for_write`], because the CSC sidecar builder in
/// `scx-format-io` needs the same rule and sits *below* this crate. Keeping the
/// name here leaves this crate's eight call sites and their tests untouched.
pub(crate) fn widest_value_encoding(encs: &[ValueEncoding]) -> ValueEncoding {
    ValueEncoding::widest_for_write(encs)
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

    /// `encode_values` is a speed change and nothing else: the bytes must equal
    /// what the per-value loop it replaces would have written, for every
    /// encoding, including into a buffer that already holds a previous row.
    #[test]
    fn encode_values_matches_the_per_value_loop() {
        let cases: &[(ValueEncoding, &[f32])] = &[
            (ValueEncoding::Uint8, &[0.0, 1.0, 7.9, 255.0]),
            (ValueEncoding::Uint16, &[0.0, 300.0, 65535.0]),
            (
                ValueEncoding::Uint32,
                &[0.0, 4_294_967_040.0, (1u64 << 32) as f32],
            ),
            (ValueEncoding::Float32, &[0.0, -1.5, f32::NAN]),
            (ValueEncoding::Float16, &[0.0, -1.5, 70000.0]),
        ];
        for &(enc, data) in cases {
            let mut want = vec![0xC3u8; 2];
            for &v in data {
                encode_value(&mut want, v, enc).unwrap();
            }
            let mut got = vec![0xC3u8; 2];
            encode_values(&mut got, data, enc).unwrap();
            assert_eq!(got, want, "{enc:?}");
        }
    }

    /// The whole reason this crate keeps its own wrapper: `OpsError::
    /// ValueOutOfRange`'s `value` / `encoding` / `max` fields are what pyscx
    /// maps and what the sibling tests above assert on, so the batch path must
    /// surface *this* crate's error rather than `scx-codec`'s. It carries the
    /// same three fields now, so lifting it is a one-variant `map_err` — but
    /// this test is what says the offending value survives that lift, and that
    /// the accumulator is left untouched, since the row that failed wrote
    /// nothing.
    #[test]
    fn encode_values_reports_the_ops_error_naming_the_offender() {
        for (enc, offender, name) in [
            (ValueEncoding::Uint8, 256.0f32, "Uint8"),
            (ValueEncoding::Uint16, 65536.0f32, "Uint16"),
            (ValueEncoding::Uint32, 8_589_934_592.0f32, "Uint32"),
            (ValueEncoding::Uint8, f32::NAN, "Uint8"),
        ] {
            for pos in 0..3 {
                let mut data = vec![1.0f32, 2.0, 3.0];
                data[pos] = offender;
                let mut buf = vec![0x11u8; 4];
                match encode_values(&mut buf, &data, enc) {
                    Err(OpsError::ValueOutOfRange {
                        value, encoding, ..
                    }) => {
                        assert_eq!(encoding, name, "{enc:?} named the wrong encoding");
                        assert!(
                            value.to_bits() == offender.to_bits(),
                            "{enc:?} reported {value} instead of the offender {offender}"
                        );
                    }
                    other => {
                        panic!("{enc:?} at index {pos}: expected ValueOutOfRange, got {other:?}")
                    }
                }
                assert_eq!(
                    buf,
                    vec![0x11u8; 4],
                    "{enc:?} left bytes behind at index {pos}"
                );
            }
        }
    }

    /// compact / merge / sort decode a shard to f32 and re-encode it under the
    /// input's own encoding. An on-disk `u32::MAX` decodes to exactly 2³²
    /// (`u32::MAX as f32` rounds up), so this arm must accept that value and
    /// saturate it back — rejecting it aborts those ops on a format-valid
    /// archive. On the detect path, `detect_value_encoding` is what keeps fresh
    /// out-of-range data away from `Uint32` in the first place; callers passing
    /// an explicit encoding bypass it and still saturate (documented on
    /// `ValueEncoding::encode_f32`).
    #[test]
    fn encode_value_uint32_preserves_decoded_u32_max() {
        let decoded_max = u32::MAX as f32;
        assert_eq!(decoded_max, (1u128 << 32) as f32);

        let mut buf = Vec::new();
        encode_value(&mut buf, decoded_max, ValueEncoding::Uint32).unwrap();
        assert_eq!(buf, u32::MAX.to_le_bytes());

        buf.clear();
        encode_value(&mut buf, 4_294_967_040.0, ValueEncoding::Uint32).unwrap();
        assert_eq!(buf, 4_294_967_040u32.to_le_bytes());

        // Genuinely out of range is still refused.
        assert!(matches!(
            encode_value(&mut Vec::new(), 8_589_934_592.0, ValueEncoding::Uint32),
            Err(OpsError::ValueOutOfRange { .. })
        ));
    }
}
