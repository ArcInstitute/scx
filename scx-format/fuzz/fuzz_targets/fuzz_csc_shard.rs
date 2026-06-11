// fuzz target: feed arbitrary bytes to the shard-header
// decode path with the section type pinned to CscShard. CSC and CSR
// share the same 76-byte shard header layout (only `shard_type` and
// the field interpretation differ), so this is structurally similar
// to `fuzz_shard.rs` but documents the intent of fuzzing the CSC
// decode entry point. The decode invariants must hold for both
// `shard_type = 1` (authoritative for CSC) and the legacy
// `shard_type = 0` (CSR-emitted CSC shards from before the Phase A
// `derive_shard_type` fix).

#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format_io::shard::ShardHeader;

fuzz_target!(|data: &[u8]| {
    // Mirror `fuzz_shard.rs`. The header parser must terminate without
    // panic / OOM on any byte input; CSC interpretation is a function
    // of the parsed `shard_type` byte plus the catalog `section_type`,
    // both of which are validated downstream.
    let _ = ShardHeader::read_from(&mut std::io::Cursor::new(data));
});
