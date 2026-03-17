#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format::catalog::FullCatalog;

fuzz_target!(|data: &[u8]| {
    let _ = FullCatalog::read_from(&mut std::io::Cursor::new(data), data.len());
});
