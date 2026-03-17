#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_codec::forbp::forbp_decode;

fuzz_target!(|data: &[u8]| {
    // Try decoding as u16 and u32 index dtypes with various row counts
    let _ = forbp_decode(data, 10, true);
    let _ = forbp_decode(data, 10, false);
    let _ = forbp_decode(data, 1, true);
    let _ = forbp_decode(data, 1, false);
    let _ = forbp_decode(data, 128, true);
});
