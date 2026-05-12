// Bitstream primitives: LSB-first bit reader/writer (docs/codec.md)

/// Error returned when a `BitReader` attempts to read past the end of its data.
#[derive(Debug, thiserror::Error)]
#[error("unexpected end of bitstream")]
pub struct BitStreamError;

// ---------------------------------------------------------------------------
// BitWriter
// ---------------------------------------------------------------------------

/// Writes bits in LSB-first order into a growable byte buffer.
pub struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_pos: u8, // 0–7: next bit position to write within current_byte
}

impl BitWriter {
    /// Create a new, empty `BitWriter`.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Write a single bit (LSB-first).
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << self.bit_pos;
        }
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Write the lowest `n_bits` bits of `value`, LSB first.
    #[inline]
    pub fn write_bits(&mut self, value: u64, n_bits: u8) {
        for i in 0..n_bits {
            self.write_bit((value >> i) & 1 != 0);
        }
    }

    /// Write a unary code: `q` ones followed by a zero.
    #[inline]
    pub fn write_unary(&mut self, q: u64) {
        for _ in 0..q {
            self.write_bit(true);
        }
        self.write_bit(false);
    }

    /// Pad the current byte to a byte boundary (zero-fill remaining bits)
    /// without consuming the writer. Needed by Rice encoder between blocks.
    pub fn pad_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Flush the writer, returning the underlying byte buffer.
    /// If there is a partial byte, it is zero-padded and pushed.
    pub fn flush(mut self) -> Vec<u8> {
        self.pad_to_byte();
        self.buffer
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BitReader — u64-buffered for high-throughput bit extraction
// ---------------------------------------------------------------------------

/// Reads bits in LSB-first order from a byte slice.
///
/// Uses a 64-bit buffer to extract multiple values per refill, eliminating
/// per-value inner loops. The hot path (`read_bits`) is a single mask+shift
/// when the buffer has enough bits.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Buffered bits, LSB-aligned. The lowest `bits_left` bits are valid.
    bit_buf: u64,
    /// Number of valid bits remaining in `bit_buf` (0–64).
    bits_left: u8,
    /// Next byte position to read from `data` into the buffer.
    byte_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` over the given data.
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            bit_buf: 0,
            bits_left: 0,
            byte_pos: 0,
        };
        reader.refill();
        reader
    }

    /// Load bytes into `bit_buf` until we have at least 56 bits (or data is
    /// exhausted). We keep a 56-bit threshold so that a single `read_bits`
    /// call for up to 56 bits never needs a mid-read refill.
    #[inline]
    fn refill(&mut self) {
        // Only refill when we have room for at least one byte
        if self.bits_left > 56 {
            return;
        }
        let remaining = self.data.len() - self.byte_pos;
        if remaining == 0 {
            return;
        }
        // Calculate how many whole bytes we can load (max space in buffer)
        let space_bytes = ((64 - self.bits_left) / 8) as usize;
        let to_load = remaining.min(space_bytes);

        // Fast path: load via u64 when we have enough data and buffer space
        if to_load >= 7 && self.bits_left == 0 {
            // Buffer is empty, load 7 bytes (56 bits) directly
            // This avoids overflowing the u64 while loading maximum data
            let mut bytes = [0u8; 8];
            bytes[..7].copy_from_slice(&self.data[self.byte_pos..self.byte_pos + 7]);
            self.bit_buf = u64::from_le_bytes(bytes);
            self.byte_pos += 7;
            self.bits_left = 56;
            return;
        }

        // General path: load bytes into buffer
        for _ in 0..to_load {
            let byte = self.data[self.byte_pos] as u64;
            self.bit_buf |= byte << self.bits_left;
            self.byte_pos += 1;
            self.bits_left += 8;
        }
    }

    /// Read a single bit (LSB-first).
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool, BitStreamError> {
        if self.bits_left == 0 {
            self.refill();
            if self.bits_left == 0 {
                return Err(BitStreamError);
            }
        }
        let bit = self.bit_buf & 1 != 0;
        self.bit_buf >>= 1;
        self.bits_left -= 1;
        Ok(bit)
    }

    /// Read `n_bits` bits and return them as a `u64`, LSB first.
    ///
    /// This is the hot path for FOR-BP and Rice decoders. With the u64 buffer,
    /// this is a single mask+shift operation when the buffer has enough bits.
    #[inline]
    pub fn read_bits(&mut self, n_bits: u8) -> Result<u64, BitStreamError> {
        // `n_bits` is a `u8`, so >64 is representable. Everything downstream
        // assumes ≤ 64 (we store the accumulator in a `u64`); a larger value
        // would underflow `bits_left -= n_bits` at the end of this function
        // and silently corrupt the bitstream position. Reject at runtime —
        // a corrupt on-disk `k` parameter would otherwise advance the
        // bitstream position incorrectly and produce wrong values without
        // erroring.
        if n_bits > 64 {
            return Err(BitStreamError);
        }
        if n_bits == 0 {
            return Ok(0);
        }
        // Ensure we have enough bits in the buffer
        if self.bits_left < n_bits {
            self.refill();
            if self.bits_left < n_bits {
                return Err(BitStreamError);
            }
        }
        // Extract n_bits from the buffer (single mask + shift)
        let mask = if n_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << n_bits) - 1
        };
        let value = self.bit_buf & mask;
        if n_bits >= 64 {
            self.bit_buf = 0;
        } else {
            self.bit_buf >>= n_bits;
        }
        self.bits_left -= n_bits;
        Ok(value)
    }

    /// Read a unary code: count ones until a zero is encountered.
    /// Returns the number of ones read.
    ///
    /// Uses `trailing_ones()` on the 64-bit buffer for fast scanning.
    #[inline]
    pub fn read_unary(&mut self) -> Result<u64, BitStreamError> {
        let mut count: u64 = 0;
        loop {
            if self.bits_left == 0 {
                self.refill();
                if self.bits_left == 0 {
                    return Err(BitStreamError);
                }
            }

            // If buffer is all zeros at this point with bits_left valid bits,
            // we need to check only the valid portion. Mask out invalid bits.
            // Since invalid high bits are already 0 (from shifts), trailing_ones
            // on bit_buf correctly counts only valid 1-bits.
            let ones = self.bit_buf.trailing_ones();

            if ones < self.bits_left as u32 {
                // Found a zero bit within the valid portion
                count += ones as u64;
                // Skip past the ones and the terminating zero
                let skip = ones + 1;
                self.bit_buf >>= skip;
                self.bits_left -= skip as u8;
                return Ok(count);
            } else {
                // All valid bits in buffer are ones — consume them all and refill
                count += self.bits_left as u64;
                self.bit_buf = 0;
                self.bits_left = 0;
            }
        }
    }

    /// Return the current position in bits from the start of the data.
    pub fn position(&self) -> usize {
        self.byte_pos * 8 - self.bits_left as usize
    }

    /// Advance to the next byte boundary. If already aligned, this is a no-op.
    pub fn align_to_byte(&mut self) {
        let pos = self.position();
        let remainder = pos % 8;
        if remainder > 0 {
            let skip = 8 - remainder;
            if skip as u8 <= self.bits_left {
                self.bit_buf >>= skip;
                self.bits_left -= skip as u8;
            } else {
                // Need more data — discard buffer and advance byte_pos
                self.bit_buf = 0;
                self.bits_left = 0;
                // byte_pos is already past the buffered data, so position()
                // is at a byte boundary after clearing the buffer
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 2.3: Round-trip random bit sequences
    #[test]
    fn round_trip_random_bits() {
        // Use a simple PRNG to avoid needing the rand crate
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut bits = Vec::with_capacity(1000);
        for _ in 0..1000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bits.push(state & 1 != 0);
        }

        let mut writer = BitWriter::new();
        for &b in &bits {
            writer.write_bit(b);
        }
        let data = writer.flush();

        let mut reader = BitReader::new(&data);
        for (i, &expected) in bits.iter().enumerate() {
            let got = reader
                .read_bit()
                .unwrap_or_else(|_| panic!("failed at bit {i}"));
            assert_eq!(got, expected, "mismatch at bit {i}");
        }
    }

    // 2.4: Edge cases
    #[test]
    fn empty_stream() {
        let writer = BitWriter::new();
        let data = writer.flush();
        assert!(data.is_empty());

        let mut reader = BitReader::new(&data);
        assert!(reader.read_bit().is_err());
    }

    #[test]
    fn single_bit_true() {
        let mut writer = BitWriter::new();
        writer.write_bit(true);
        let data = writer.flush();
        assert_eq!(data, vec![0x01]);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bit().unwrap(), true);
    }

    #[test]
    fn single_bit_false() {
        let mut writer = BitWriter::new();
        writer.write_bit(false);
        let data = writer.flush();
        assert_eq!(data, vec![0x00]);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bit().unwrap(), false);
    }

    #[test]
    fn byte_aligned_write() {
        // Write exactly 8 bits: 0b10110011 = 0xB3
        let mut writer = BitWriter::new();
        writer.write_bits(0xB3, 8);
        let data = writer.flush();
        assert_eq!(data, vec![0xB3]);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(8).unwrap(), 0xB3);
    }

    #[test]
    fn full_64_bit_value() {
        let val: u64 = 0xFEDC_BA98_7654_3210;
        let mut writer = BitWriter::new();
        writer.write_bits(val, 64);
        let data = writer.flush();
        assert_eq!(data.len(), 8);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(64).unwrap(), val);
    }

    // 2.5: Unary encode/decode
    #[test]
    fn unary_values() {
        for &q in &[0u64, 1, 2, 100, 1000] {
            let mut writer = BitWriter::new();
            writer.write_unary(q);
            let data = writer.flush();

            let mut reader = BitReader::new(&data);
            assert_eq!(reader.read_unary().unwrap(), q, "unary mismatch for q={q}");
        }
    }

    // 2.6: Mixed operations
    #[test]
    fn mixed_operations() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b10110, 5); // 5-bit value
        writer.write_unary(3); // 1110
        writer.write_bit(true); // 1
        writer.write_bits(0xFF, 8); // 8-bit value
        writer.write_unary(0); // 0
        let data = writer.flush();

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(5).unwrap(), 0b10110);
        assert_eq!(reader.read_unary().unwrap(), 3);
        assert_eq!(reader.read_bit().unwrap(), true);
        assert_eq!(reader.read_bits(8).unwrap(), 0xFF);
        assert_eq!(reader.read_unary().unwrap(), 0);
    }

    // 2.7: Boundary — write 7 bits, then 1 bit (completes byte), then 3 more
    #[test]
    fn boundary_crossing() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1010101, 7); // 7 bits: 1010101
        writer.write_bit(true); // completes first byte
        writer.write_bits(0b110, 3); // 3 bits into second byte
        let data = writer.flush();
        assert_eq!(data.len(), 2);

        // First byte: bits 0-6 = 1010101, bit 7 = 1 → 0b11010101 = 0xD5
        assert_eq!(data[0], 0xD5);
        // Second byte: bits 0-2 = 110 (rest zero-padded) → 0b00000110 = 0x06
        assert_eq!(data[1], 0x06);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(7).unwrap(), 0b1010101);
        assert_eq!(reader.read_bit().unwrap(), true);
        assert_eq!(reader.read_bits(3).unwrap(), 0b110);
    }

    // Read past end returns error
    #[test]
    fn read_past_end() {
        let data = vec![0xFF];
        let mut reader = BitReader::new(&data);
        // Read all 8 bits
        for _ in 0..8 {
            reader.read_bit().unwrap();
        }
        // Next read should fail
        assert!(reader.read_bit().is_err());
    }

    #[test]
    fn read_bits_past_end() {
        let data = vec![0xFF];
        let mut reader = BitReader::new(&data);
        reader.read_bits(4).unwrap();
        // Only 4 bits left, asking for 8 should fail
        assert!(reader.read_bits(8).is_err());
    }

    #[test]
    fn pad_to_byte_and_continue() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.pad_to_byte();
        writer.write_bits(0xFF, 8);
        let data = writer.flush();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], 0b00000101);
        assert_eq!(data[1], 0xFF);
    }

    #[test]
    fn align_to_byte_reader() {
        let data = vec![0xFF, 0xAB];
        let mut reader = BitReader::new(&data);
        reader.read_bits(3).unwrap(); // read 3 bits from first byte
        reader.align_to_byte(); // skip remaining 5 bits
        assert_eq!(reader.position(), 8);
        assert_eq!(reader.read_bits(8).unwrap(), 0xAB);
    }

    #[test]
    fn position_tracking() {
        let data = vec![0xFF, 0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.position(), 0);
        reader.read_bit().unwrap();
        assert_eq!(reader.position(), 1);
        reader.read_bits(5).unwrap();
        assert_eq!(reader.position(), 6);
        reader.read_bits(2).unwrap();
        assert_eq!(reader.position(), 8);
    }

    #[test]
    fn write_zero_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xDEAD, 0); // should be a no-op
        writer.write_bits(0b11, 2);
        let data = writer.flush();
        assert_eq!(data, vec![0b11]);
    }
}
