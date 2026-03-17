#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format::header::FileHeader;

fuzz_target!(|data: &[u8]| {
    let _ = FileHeader::read_from(&mut std::io::Cursor::new(data));
});
