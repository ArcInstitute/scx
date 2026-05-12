//! Byte-shuffle pre-filter for LZ4 codec (Phase 2D, SPEC W7).
//!
//! Reorders an array of N elements of width W bytes so that all byte-0 values
//! are contiguous, then all byte-1, etc. This is a transpose of an N x W matrix
//! and dramatically improves compression of typed arrays by grouping similar
//! bytes together.

use crate::dispatch::CodecError;

/// Byte-shuffle: transpose N elements of `element_width` bytes.
///
/// For `element_width == 1` (u8 values), returns a copy unchanged.
/// For `element_width == 0` or empty input, returns empty.
///
/// Returns `Err(CodecError::MalformedInput)` if `input.len()` is not an exact
/// multiple of `element_width` — a malformed or corrupt input would otherwise
/// silently drop trailing bytes and break round-trip.
pub fn byte_shuffle(input: &[u8], element_width: usize) -> Result<Vec<u8>, CodecError> {
    if element_width <= 1 || input.is_empty() {
        return Ok(input.to_vec());
    }
    if !input.len().is_multiple_of(element_width) {
        return Err(CodecError::MalformedInput(format!(
            "byte_shuffle: input length {} not divisible by element_width {}",
            input.len(),
            element_width
        )));
    }
    let n = input.len() / element_width;
    let mut out = vec![0u8; input.len()];
    for i in 0..n {
        for b in 0..element_width {
            out[b * n + i] = input[i * element_width + b];
        }
    }
    Ok(out)
}

/// Byte-unshuffle: inverse transpose of N elements of `element_width` bytes.
///
/// For `element_width == 1` (u8 values), returns a copy unchanged.
/// For `element_width == 0` or empty input, returns empty.
///
/// Returns `Err(CodecError::MalformedInput)` if `input.len()` is not an exact
/// multiple of `element_width`. This is the decoder-facing direction, so a
/// corrupt on-disk shard would otherwise silently produce truncated output.
pub fn byte_unshuffle(input: &[u8], element_width: usize) -> Result<Vec<u8>, CodecError> {
    if element_width <= 1 || input.is_empty() {
        return Ok(input.to_vec());
    }
    if !input.len().is_multiple_of(element_width) {
        return Err(CodecError::MalformedInput(format!(
            "byte_unshuffle: input length {} not divisible by element_width {}",
            input.len(),
            element_width
        )));
    }
    let n = input.len() / element_width;
    let mut out = vec![0u8; input.len()];
    for i in 0..n {
        for b in 0..element_width {
            out[i * element_width + b] = input[b * n + i];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_unshuffle_roundtrip_u32() {
        // 3 u32 values: [0x04030201, 0x08070605, 0x0C0B0A09]
        let input: Vec<u8> = vec![
            0x01, 0x02, 0x03, 0x04, // element 0
            0x05, 0x06, 0x07, 0x08, // element 1
            0x09, 0x0A, 0x0B, 0x0C, // element 2
        ];
        let shuffled = byte_shuffle(&input, 4).unwrap();
        // After shuffle: byte-0 of each element, then byte-1, etc.
        assert_eq!(
            shuffled,
            vec![
                0x01, 0x05, 0x09, // byte-0 lane
                0x02, 0x06, 0x0A, // byte-1 lane
                0x03, 0x07, 0x0B, // byte-2 lane
                0x04, 0x08, 0x0C, // byte-3 lane
            ]
        );
        let unshuffled = byte_unshuffle(&shuffled, 4).unwrap();
        assert_eq!(unshuffled, input);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip_u16() {
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let shuffled = byte_shuffle(&input, 2).unwrap();
        assert_eq!(shuffled, vec![0x01, 0x03, 0x05, 0x02, 0x04, 0x06]);
        assert_eq!(byte_unshuffle(&shuffled, 2).unwrap(), input);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip_u64() {
        let input: Vec<u8> = (0..24).collect();
        let shuffled = byte_shuffle(&input, 8).unwrap();
        let unshuffled = byte_unshuffle(&shuffled, 8).unwrap();
        assert_eq!(unshuffled, input);
    }

    #[test]
    fn shuffle_u8_is_identity() {
        let input = vec![1u8, 2, 3, 4, 5];
        assert_eq!(byte_shuffle(&input, 1).unwrap(), input);
        assert_eq!(byte_unshuffle(&input, 1).unwrap(), input);
    }

    #[test]
    fn shuffle_empty() {
        let empty: Vec<u8> = vec![];
        assert_eq!(byte_shuffle(&empty, 4).unwrap(), empty);
        assert_eq!(byte_unshuffle(&empty, 4).unwrap(), empty);
    }

    #[test]
    fn shuffle_ragged_input_errors() {
        // 7 bytes is not divisible by element_width 4.
        let input: Vec<u8> = vec![0; 7];
        assert!(matches!(
            byte_shuffle(&input, 4),
            Err(CodecError::MalformedInput(_))
        ));
        assert!(matches!(
            byte_unshuffle(&input, 4),
            Err(CodecError::MalformedInput(_))
        ));
    }
}
