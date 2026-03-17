#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_codec::rice::{rice_decode, B_VAL};

fuzz_target!(|data: &[u8]| {
    // Try decoding arbitrary bytes as Rice-encoded values with various counts
    let _ = rice_decode(data, 256, B_VAL);
    let _ = rice_decode(data, 10, B_VAL);
    let _ = rice_decode(data, 1, B_VAL);
    let _ = rice_decode(data, 0, B_VAL);

    // Try with a small block size
    let _ = rice_decode(data, 5, 4);
});
