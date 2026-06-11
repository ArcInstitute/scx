// Fuzz target: feed arbitrary bytes to `ModalityTable::read_from`.
// The decode path must terminate without panic or OOM on any input,
// regardless of magic, version, entry count, name lengths, or modality
// type discriminants.
//
// Mirrors `fuzz_bitmap.rs` / `fuzz_csc_shard.rs` style — a single-line
// read invocation with the section length set to the input buffer length.

#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format_io::ModalityTable;

fuzz_target!(|data: &[u8]| {
    let _ = ModalityTable::read_from(&mut std::io::Cursor::new(data), data.len());
});
