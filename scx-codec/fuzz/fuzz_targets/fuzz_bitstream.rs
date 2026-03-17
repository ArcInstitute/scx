#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_codec::bitstream::BitReader;

fuzz_target!(|data: &[u8]| {
    // Try reading bits — should never panic on arbitrary input
    {
        let mut reader = BitReader::new(data);
        while reader.read_bit().is_ok() {}
    }

    // Try reading unary codes
    {
        let mut reader = BitReader::new(data);
        while reader.read_unary().is_ok() {}
    }

    // Try reading 8-bit values
    {
        let mut reader = BitReader::new(data);
        while reader.read_bits(8).is_ok() {}
    }

    // Try reading mixed widths
    {
        let mut reader = BitReader::new(data);
        for width in [1, 3, 5, 7, 8, 16, 32, 64].iter().cycle() {
            if reader.read_bits(*width).is_err() {
                break;
            }
        }
    }
});
