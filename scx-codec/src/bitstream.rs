// Bitstream primitives: LSB-first bit reader/writer (SPEC §4)

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
// BitReader
// ---------------------------------------------------------------------------

/// Reads bits in LSB-first order from a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0–7: next bit position to read within data[byte_pos]
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` over the given data.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit (LSB-first).
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool, BitStreamError> {
        if self.byte_pos >= self.data.len() {
            return Err(BitStreamError);
        }
        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1 != 0;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        Ok(bit)
    }

    /// Read `n_bits` bits and return them as a `u64`, LSB first.
    #[inline]
    pub fn read_bits(&mut self, n_bits: u8) -> Result<u64, BitStreamError> {
        let mut value: u64 = 0;
        for i in 0..n_bits {
            if self.read_bit()? {
                value |= 1u64 << i;
            }
        }
        Ok(value)
    }

    /// Read a unary code: count ones until a zero is encountered.
    /// Returns the number of ones read.
    #[inline]
    pub fn read_unary(&mut self) -> Result<u64, BitStreamError> {
        let mut count: u64 = 0;
        loop {
            if self.read_bit()? {
                count += 1;
            } else {
                return Ok(count);
            }
        }
    }

    /// Return the current position in bits.
    pub fn position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Advance to the next byte boundary. If already aligned, this is a no-op.
    pub fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (Phase1 tasks 2.3–2.7)
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
