#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // TODO: fuzz FileHeader::read with arbitrary bytes
    let _ = data;
});
