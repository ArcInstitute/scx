#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format::shard::ShardHeader;

fuzz_target!(|data: &[u8]| {
    let _ = ShardHeader::read_from(&mut std::io::Cursor::new(data));
});
