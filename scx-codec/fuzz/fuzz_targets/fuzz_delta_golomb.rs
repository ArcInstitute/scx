#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_codec::delta_golomb::delta_golomb_decode;

fuzz_target!(|data: &[u8]| {
    // Try decoding with various expected lengths
    let _ = delta_golomb_decode(data, 10);
    let _ = delta_golomb_decode(data, 1);
    let _ = delta_golomb_decode(data, 100);
    let _ = delta_golomb_decode(data, 0);
});
