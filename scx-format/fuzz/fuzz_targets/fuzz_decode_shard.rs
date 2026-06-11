// Fuzz the full CSR shard decode path (header parse + bounds-checked payload
// slicing + codec dispatch), not just the header parse that `fuzz_shard`
// covers. The catalog BLAKE3 authenticates catalog bytes, not shard payloads,
// so `decode_shard_bytes` must terminate with a `Result` — never panic or OOM —
// on arbitrary bytes pointing arbitrary internal offsets/lengths (finding F1).

#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_format_io::{decode_shard_bytes, FullCatalogEntry, SectionType};

fuzz_target!(|data: &[u8]| {
    let entry = FullCatalogEntry {
        name: "X_shard_0".to_string(),
        offset: 0,
        length: data.len() as u64,
        section_type: SectionType::CsrShard,
        checksum: [0u8; 32],
        modality_id: 0,
        stats: None,
    };
    // catalog_version = 2 exercises the strict shard-type validation;
    // verify_checksum = false so the codec path is reached regardless of bytes.
    let _ = decode_shard_bytes(data, &entry, 2, false);
});
