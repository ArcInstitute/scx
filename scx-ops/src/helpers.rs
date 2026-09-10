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
            // Inclusive at 2³² — see `ValueEncoding::encode_f32`, which this
            // mirrors. `u32::MAX as f32` IS 2³², so this value is exactly what
            // an on-disk `u32::MAX` decodes to; compact/merge/sort re-encode
            // decoded f32 under the input's own encoding, and rejecting it
            // would abort them on format-valid archives. `as u32` saturates
            // back to `u32::MAX`, restoring the original value.
            //
            // Fresh out-of-range data is diverted to `Float32` by
            // `detect_value_encoding`, on the detect path and on
            // `attach_external_layer`'s post-canonicalization one.
            // `encoding_for_canonicalized` deliberately does *not* divert at
            // exactly 2³² — its values may be decoded originals, and saturating
            // is what restores them — so the rewrite ops reach this arm by
            // design. A caller passing an explicit encoding bypasses every
            // detector; so does any `u32` above 2²⁴, which the f32 decode
            // rounded long before reaching here. See `ValueEncoding::encode_f32`.
            const UINT32_BOUND: f32 = (1u128 << 32) as f32;
            if !(0.0..=UINT32_BOUND).contains(&value) {
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

/// [`encode_value`] over a whole contiguous run of values.
///
/// The rewrite ops' accumulator loops call this once per kept row rather than
/// once per nonzero. Measured on `scx compact --codec auto` at census_1m,
/// `encode_value` was **5.08 % of all cycles**, and because it runs on the
/// calling thread while the encode runs on the pool, that is roughly **16 % of
/// the op's wall** — the largest single serial term left after PR-42.
///
/// The fast path is [`ValueEncoding::encode_f32_into`], so there is one
/// width-specialised implementation in the workspace rather than two — 1.7-2.1x
/// on the integer widths and 27.8x on `Float32`, benched there. That is short of
/// the 5-10x the OPT plan projected for this pass, so expect roughly 8 % off a
/// census_1m compact from it, not 14 %. What this
/// wrapper owns is the *error*: `scx-codec` reports an out-of-range value as a
/// `CodecError::Io` carrying only a message, while this crate's
/// [`OpsError::ValueOutOfRange`] carries `value` / `encoding` / `max` — which
/// pyscx maps to a Python exception and three tests assert on. So on rejection
/// it replays the run through [`encode_value`] to raise this crate's error.
///
/// `buf` is left exactly as it was found on error: an accumulator must not gain
/// half a row behind an `Err`, because compact's `widest_value_encoding` retry
/// story assumes the failed row wrote nothing.
pub fn encode_values(
    buf: &mut Vec<u8>,
    data: &[f32],
    encoding: ValueEncoding,
) -> crate::error::Result<()> {
    let start = buf.len();
    if encoding.encode_f32_into(buf, data).is_ok() {
        return Ok(());
    }
    buf.truncate(start);
    for &v in data {
        if let Err(e) = encode_value(buf, v, encoding) {
            buf.truncate(start);
            return Err(e);
        }
    }
    // Reached only if the two crates' range checks disagree — they are the same
    // bounds arm for arm today, and `Uint32`'s is a shared constant. If one ever
    // loosens, this crate's answer is the one the ops contract is written
    // against, so a run `encode_value` accepts is a success, not an error.
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

    /// The whole reason this crate keeps its own wrapper: `scx-codec` reports an
    /// out-of-range value as a `CodecError::Io` carrying only a message, and
    /// `OpsError::ValueOutOfRange`'s `value` / `encoding` / `max` fields are
    /// what pyscx maps and what the sibling tests above assert on. So the batch
    /// path must surface *this* crate's error, with the offending value in it —
    /// and must leave the accumulator untouched, since the row that failed
    /// wrote nothing.
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
