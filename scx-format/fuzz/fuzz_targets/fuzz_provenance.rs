// Fuzz target: feed arbitrary bytes to `Provenance::read_from`.
// The `params_json` blob is a meaningful trust boundary across the
// convert/append/delete/rollback phases — the parser must terminate
// without panic or OOM on any input regardless of length prefixes,
// UTF-8 validity, or entry counts.
//
// Mirrors `fuzz_bitmap.rs` / `fuzz_modality_table.rs` style — a
// single-line read invocation with the section length set to the input
// buffer length.

#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format_io::Provenance;

fuzz_target!(|data: &[u8]| {
    let _ = Provenance::read_from(&mut std::io::Cursor::new(data), data.len());
});
