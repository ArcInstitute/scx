// Read-only summaries over a file's per-shard headers.
//
// The per-shard `codec_id` is the only record of what an adaptive
// `codec="auto"` write actually chose (the file header's `codec_id` is a hint
// and mutating ops write 0 there). Lives here rather than in `scx-cli` because
// `scx-cli` is a bin-only crate: `scx-ops` tests, the CLI integration tests and
// pyscx all need this and none of them can import from a binary.

use std::collections::BTreeMap;

use crate::catalog::FullCatalogEntry;
use crate::reader::ScxReader;
use crate::shard::ShardHeader;

/// Per-`codec_id` shard count across `shards`, sorted ascending by codec id.
///
/// Surfaces what an adaptive `codec="auto"` write actually chose per shard
/// (predominantly ShufDeltaZstd, mixed with Scx1 for low-median shards) rather
/// than a single "the codec". Reads each shard's 76-byte header.
pub fn codec_id_histogram(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
) -> Result<Vec<(u8, u64)>, crate::error::ScxError> {
    let mut counts: BTreeMap<u8, u64> = BTreeMap::new();
    for e in shards {
        let h = reader.read_shard_header(e)?;
        *counts.entry(h.codec_id).or_insert(0) += 1;
    }
    Ok(counts.into_iter().collect())
}

/// Read every shard header in `shards`, project one byte field via `extract`,
/// and return the distinct values sorted ascending.
pub fn distinct_sorted_shard_field<F>(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    extract: F,
) -> Result<Vec<u8>, crate::error::ScxError>
where
    F: Fn(&ShardHeader) -> u8,
{
    let mut bytes: Vec<u8> = shards
        .iter()
        .map(|e| reader.read_shard_header(e).map(|h| extract(&h)))
        .collect::<Result<Vec<_>, _>>()?;
    bytes.sort_unstable();
    bytes.dedup();
    Ok(bytes)
}

/// Total encoded bytes of every shard in `shards` (header + payload), read from
/// the catalog rather than by decoding.
///
/// The size half of a codec-regression assertion: a codec flip shows up as both
/// a changed `codec_id` histogram and a changed byte total, and asserting only
/// the former would miss a world where framing overhead ate the win.
pub fn total_shard_bytes(shards: &[&FullCatalogEntry]) -> u64 {
    shards.iter().map(|e| e.length).sum()
}
