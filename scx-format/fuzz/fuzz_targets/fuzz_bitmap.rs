// Fuzz target: feed arbitrary bytes to `BitmapShard::read_from`, then exercise
// `per_gene_counts()` on any successfully-parsed shard. The decode path must
// terminate without panic or OOM on any input, regardless of magic, version,
// orientation, gene_id widths, or roaring-bitmap payload shape; and a parsed
// shard's gene_ids must stay in `0..n_vars` so `per_gene_counts` cannot index
// out of bounds (semantic-range finding F3 — parse rejects out-of-range
// gene_ids, this drives the post-parse consumer to prove it).

#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format::BitmapShard;

fuzz_target!(|data: &[u8]| {
    if let Ok(shard) = BitmapShard::read_from(&mut std::io::Cursor::new(data), data.len()) {
        let _ = shard.per_gene_counts();
    }
});
