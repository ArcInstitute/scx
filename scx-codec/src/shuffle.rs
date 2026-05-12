/// Byte-shuffle pre-filter for LZ4 codec (Phase 2D, SPEC W7).
///
/// Reorders an array of N elements of width W bytes so that all byte-0 values
/// are contiguous, then all byte-1, etc. This is a transpose of an N x W matrix
/// and dramatically improves compression of typed arrays by grouping similar
/// bytes together.
///
/// Byte-shuffle: transpose N elements of `element_width` bytes.
///
/// For `element_width == 1` (u8 values), returns a copy unchanged.
/// For `element_width == 0` or empty input, returns empty.
pub fn byte_shuffle(input: &[u8], element_width: usize) -> Vec<u8> {
    if element_width <= 1 || input.is_empty() {
        return input.to_vec();
    }
    let n = input.len() / element_width;
    debug_assert_eq!(
        input.len() % element_width,
        0,
        "input length {} not divisible by element_width {}",
        input.len(),
        element_width
    );
    let mut out = vec![0u8; input.len()];
    for i in 0..n {
        for b in 0..element_width {
            out[b * n + i] = input[i * element_width + b];
        }
    }
    out
}

/// Byte-unshuffle: inverse transpose of N elements of `element_width` bytes.
///
/// For `element_width == 1` (u8 values), returns a copy unchanged.
/// For `element_width == 0` or empty input, returns empty.
pub fn byte_unshuffle(input: &[u8], element_width: usize) -> Vec<u8> {
    if element_width <= 1 || input.is_empty() {
        return input.to_vec();
    }
    let n = input.len() / element_width;
    debug_assert_eq!(
        input.len() % element_width,
        0,
        "input length {} not divisible by element_width {}",
        input.len(),
        element_width
    );
    let mut out = vec![0u8; input.len()];
    for i in 0..n {
        for b in 0..element_width {
            out[i * element_width + b] = input[b * n + i];
        }
    }
    out
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
        let shuffled = byte_shuffle(&input, 4);
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
        let unshuffled = byte_unshuffle(&shuffled, 4);
        assert_eq!(unshuffled, input);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip_u16() {
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let shuffled = byte_shuffle(&input, 2);
        assert_eq!(shuffled, vec![0x01, 0x03, 0x05, 0x02, 0x04, 0x06]);
        assert_eq!(byte_unshuffle(&shuffled, 2), input);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip_u64() {
        let input: Vec<u8> = (0..24).collect();
        let shuffled = byte_shuffle(&input, 8);
        let unshuffled = byte_unshuffle(&shuffled, 8);
        assert_eq!(unshuffled, input);
    }

    #[test]
    fn shuffle_u8_is_identity() {
        let input = vec![1u8, 2, 3, 4, 5];
        assert_eq!(byte_shuffle(&input, 1), input);
        assert_eq!(byte_unshuffle(&input, 1), input);
    }

    #[test]
    fn shuffle_empty() {
        let empty: Vec<u8> = vec![];
        assert_eq!(byte_shuffle(&empty, 4), empty);
        assert_eq!(byte_unshuffle(&empty, 4), empty);
    }
}
