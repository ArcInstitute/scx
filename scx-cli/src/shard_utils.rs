// Shared CLI helpers for decoding shard headers.
//
// The `ValueEncoding`/`CodecId` byte → enum decode (with its error
// message) and the "read each shard header and collect a distinct,
// sorted set of one header field" idiom repeat across `query`,
// `upgrade`, and `info`. These helpers define them once.

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::reader::ScxReader;
use scx_format_io::shard::ShardHeader;

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Decode a `value_encoding` byte, erroring with the canonical message
/// on an unknown value.
pub fn decode_value_encoding(byte: u8) -> CliResult<ValueEncoding> {
    ValueEncoding::from_u8(byte).ok_or_else(|| format!("unknown value encoding: {byte}").into())
}

/// Decode a `codec_id` byte, erroring with the canonical message on an
/// unknown value.
pub fn decode_codec_id(byte: u8) -> CliResult<CodecId> {
    CodecId::from_u8(byte).ok_or_else(|| format!("unknown codec: {byte}").into())
}

/// Read every shard header in `shards`, project one byte field via
/// `extract`, and return the distinct values sorted ascending. Used by
/// `info` to summarize per-shard value-encoding / codec mixes.
///
/// Thin shim over `scx_format_io::distinct_sorted_shard_field`, which is where
/// the implementation lives so `scx-ops` tests (which cannot import this
/// bin-only crate) can share it.
pub fn distinct_sorted_shard_field<F>(
    reader: &ScxReader,
    shards: &[&scx_format_io::catalog::FullCatalogEntry],
    extract: F,
) -> CliResult<Vec<u8>>
where
    F: Fn(&ShardHeader) -> u8,
{
    Ok(scx_format_io::distinct_sorted_shard_field(
        reader, shards, extract,
    )?)
}

/// Per-`codec_id` shard count across `shards`, sorted ascending by codec id.
///
/// Surfaces what an adaptive `codec="auto"` write actually chose per shard
/// (predominantly ShufDeltaZstd, mixed with Scx1 for low-median shards) rather
/// than a single "the codec". Reads each shard's 76-byte header.
///
/// Thin shim over `scx_format_io::codec_id_histogram` — see above.
pub fn codec_id_histogram(
    reader: &ScxReader,
    shards: &[&scx_format_io::catalog::FullCatalogEntry],
) -> CliResult<Vec<(u8, u64)>> {
    Ok(scx_format_io::codec_id_histogram(reader, shards)?)
}
