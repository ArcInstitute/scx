// Phase 5b fuzz target: feed arbitrary bytes to `BitmapShard::read_from`.
// The decode path must terminate without panic or OOM on any input,
// regardless of magic, version, orientation, gene_id widths, or
// roaring-bitmap payload shape.
//
// Mirrors `fuzz_csc_shard.rs` style — a single-line read invocation
// with the section length set to the input buffer length.

#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format::BitmapShard;

fuzz_target!(|data: &[u8]| {
    let _ = BitmapShard::read_from(&mut std::io::Cursor::new(data), data.len());
});
