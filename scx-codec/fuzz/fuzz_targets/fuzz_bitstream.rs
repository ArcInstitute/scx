#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // TODO: fuzz BitReader with arbitrary bytes
    let _ = data;
});
