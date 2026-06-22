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
            if value < 0.0 || value > u8::MAX as f32 {
                return Err(OpsError::ValueOutOfRange {
                    value,
                    encoding: "Uint8",
                    max: u8::MAX as f32,
                });
            }
            buf.push(value as u8);
        }
        ValueEncoding::Uint16 => {
            if value < 0.0 || value > u16::MAX as f32 {
                return Err(OpsError::ValueOutOfRange {
                    value,
                    encoding: "Uint16",
                    max: u16::MAX as f32,
                });
            }
            buf.extend_from_slice(&(value as u16).to_le_bytes());
        }
        ValueEncoding::Uint32 => {
            if value < 0.0 || value > u32::MAX as f32 {
                return Err(OpsError::ValueOutOfRange {
                    value,
                    encoding: "Uint32",
                    max: u32::MAX as f32,
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
