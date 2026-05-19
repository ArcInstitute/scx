// Fuzz target: feed arbitrary bytes to `PredicateIndex::read_from`.
// Covers both `CategoricalIndex` and `NumericIndex` variants implicitly
// via the per-column discriminant byte. The decode path must terminate
// without panic or OOM on any input, regardless of column count,
// dictionary lengths, or B-tree page geometry.
//
// Mirrors `scx-format/fuzz/fuzz_targets/fuzz_bitmap.rs` style — a
// single-line read invocation.

#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_engine::PredicateIndex;

fuzz_target!(|data: &[u8]| {
    let _ = PredicateIndex::read_from(&mut std::io::Cursor::new(data));
});
