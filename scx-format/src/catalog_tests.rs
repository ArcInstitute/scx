use super::*;
use std::io::Cursor;

// -----------------------------------------------------------------------
// RootCatalog tests
// -----------------------------------------------------------------------

fn sample_root_entry(group_type: u8) -> RootCatalogEntry {
    RootCatalogEntry {
        group_type,
        first_section_offset: 4352,
        total_group_length: 1_000_000,
        n_sections: 4,
        summary: [0xAB; 32],
    }
}

#[test]
fn root_catalog_round_trip() {
    let catalog = RootCatalog {
        n_section_groups: 3,
        entries: vec![
            sample_root_entry(SectionType::CsrShard as u8),
            sample_root_entry(SectionType::ObsMetadata as u8),
            sample_root_entry(SectionType::VarMetadata as u8),
        ],
    };

    let mut buf = Vec::new();
    catalog.write_to(&mut buf).unwrap();

    let expected_size = 2 + 3 * ROOT_CATALOG_ENTRY_SIZE;
    assert_eq!(buf.len(), expected_size);

    let mut cursor = Cursor::new(&buf);
    let decoded = RootCatalog::read_from(&mut cursor).unwrap();
    assert_eq!(decoded, catalog);
}

#[test]
fn root_catalog_empty() {
    let catalog = RootCatalog {
        n_section_groups: 0,
        entries: vec![],
    };

    let mut buf = Vec::new();
    catalog.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), 2); // just the u16

    let mut cursor = Cursor::new(&buf);
    let decoded = RootCatalog::read_from(&mut cursor).unwrap();
    assert_eq!(decoded, catalog);
}

#[test]
fn root_catalog_too_large() {
    // 77 entries * 53 bytes + 2 = 4083; 78 entries * 53 + 2 = 4136 > 4096
    let entries: Vec<_> = (0..78).map(|i| sample_root_entry(i % 13)).collect();
    let catalog = RootCatalog {
        n_section_groups: 78,
        entries,
    };

    let mut buf = Vec::new();
    let err = catalog.write_to(&mut buf).unwrap_err();
    assert!(matches!(err, ScxError::RootCatalogTooLarge(_)));
}

#[test]
fn root_catalog_entry_size_constant() {
    let entry = sample_root_entry(0);
    let mut buf = Vec::new();
    entry.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), ROOT_CATALOG_ENTRY_SIZE);
}

// -----------------------------------------------------------------------
// ShardStats tests (9.17)
// -----------------------------------------------------------------------

fn sample_stats() -> ShardStats {
    ShardStats {
        row_start: 0,
        row_end: 16384,
        col_start: 0,
        col_end: 30_000,
        nnz: 5_000_000,
        value_min: 1,
        value_max: 65535,
        value_sum: 50_000_000,
        n_indexed_columns: 0,
        column_stats: vec![],
    }
}

#[test]
fn shard_stats_round_trip() {
    let stats = sample_stats();
    let mut buf = Vec::new();
    stats.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), SHARD_STATS_BASE_SIZE_V2);

    let mut cursor = Cursor::new(&buf);
    let decoded = ShardStats::read_from(&mut cursor, CURRENT_CATALOG_VERSION).unwrap();
    assert_eq!(decoded, stats);
}

#[test]
fn shard_stats_extreme_values() {
    let stats = ShardStats {
        row_start: u64::MAX - 1,
        row_end: u64::MAX,
        col_start: u64::MAX - 1,
        col_end: u64::MAX,
        nnz: u64::MAX,
        value_min: 0,
        value_max: u32::MAX,
        value_sum: u64::MAX,
        n_indexed_columns: 0,
        column_stats: vec![],
    };
    let mut buf = Vec::new();
    stats.write_to(&mut buf).unwrap();

    let mut cursor = Cursor::new(&buf);
    let decoded = ShardStats::read_from(&mut cursor, CURRENT_CATALOG_VERSION).unwrap();
    assert_eq!(decoded, stats);
}

#[test]
fn shard_stats_with_column_stats_round_trip() {
    let stats = ShardStats {
        row_start: 0,
        row_end: 1000,
        col_start: 0,
        col_end: 5_000,
        nnz: 5000,
        value_min: 1,
        value_max: 255,
        value_sum: 50000,
        n_indexed_columns: 2,
        column_stats: vec![
            ColumnStat::MinMax {
                column_name_hash: column_name_hash("n_genes"),
                min: 100.0,
                max: 5000.0,
            },
            ColumnStat::CategoryBitset {
                column_name_hash: column_name_hash("cell_type"),
                bitset: vec![0b0000_0101, 0b0000_0010], // categories 0, 2, 9 present
            },
        ],
    };
    let mut buf = Vec::new();
    stats.write_to(&mut buf).unwrap();
    assert!(buf.len() > SHARD_STATS_BASE_SIZE_V2); // larger than base

    let mut cursor = Cursor::new(&buf);
    let decoded = ShardStats::read_from(&mut cursor, CURRENT_CATALOG_VERSION).unwrap();
    assert_eq!(decoded.n_indexed_columns, 2);
    assert_eq!(decoded.column_stats.len(), 2);
    assert_eq!(decoded, stats);
}

#[test]
fn shard_stats_zero_indexed_backward_compat() {
    // Phase 1 file: n_indexed_columns == 0, no column_stats
    let stats = ShardStats {
        row_start: 0,
        row_end: 100,
        col_start: 0,
        col_end: 200,
        nnz: 200,
        value_min: 1,
        value_max: 10,
        value_sum: 500,
        n_indexed_columns: 0,
        column_stats: vec![],
    };
    let mut buf = Vec::new();
    stats.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), SHARD_STATS_BASE_SIZE_V2);

    let mut cursor = Cursor::new(&buf);
    let decoded = ShardStats::read_from(&mut cursor, CURRENT_CATALOG_VERSION).unwrap();
    assert_eq!(decoded, stats);
    assert!(decoded.column_stats.is_empty());
}

/// v1 catalog read path: a CSC entry with axis-overloaded
/// `row_start`/`row_end` is reconciled to populate
/// `col_start`/`col_end` after `FullCatalog::read_from` returns.
#[test]
fn v1_catalog_csc_axis_reconciliation() {
    // Hand-build a v1 catalog payload with a CSC entry whose stats
    // are encoded in the legacy 41-byte layout (axis-overloaded
    // row_start / row_end).
    let mut v1_stats_bytes = Vec::new();
    // row_start = 100, row_end = 250 (axis-overloaded col range)
    v1_stats_bytes.extend_from_slice(&100u64.to_le_bytes());
    v1_stats_bytes.extend_from_slice(&250u64.to_le_bytes());
    v1_stats_bytes.extend_from_slice(&1000u64.to_le_bytes()); // nnz
    v1_stats_bytes.extend_from_slice(&1u32.to_le_bytes()); // value_min
    v1_stats_bytes.extend_from_slice(&255u32.to_le_bytes()); // value_max
    v1_stats_bytes.extend_from_slice(&50000u64.to_le_bytes()); // value_sum
    v1_stats_bytes.push(0); // n_indexed_columns
    assert_eq!(v1_stats_bytes.len(), SHARD_STATS_BASE_SIZE_V1);

    // v1 catalog header: catalog_version=1, manifest_seq=0,
    // prev_catalog_offset=0, n_obs=1000, n_entries=1
    let mut payload = Vec::new();
    payload.extend_from_slice(&1u16.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    payload.extend_from_slice(&1000u64.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());

    // entry: name="X_csc_0", offset=4352, length=10000,
    // section_type=CscShard, checksum=0..., stats=(v1)
    let name = b"X_csc_0";
    payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
    payload.extend_from_slice(name);
    payload.extend_from_slice(&4352u64.to_le_bytes());
    payload.extend_from_slice(&10000u64.to_le_bytes());
    payload.push(SectionType::CscShard as u8);
    payload.extend_from_slice(&[0u8; 32]);
    payload.extend_from_slice(&(v1_stats_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(&v1_stats_bytes);

    // trailing 32-byte BLAKE3
    let checksum = crate::checksum::blake3_hash(&payload);
    let mut full = payload.clone();
    full.extend_from_slice(&checksum);

    let mut cur = Cursor::new(&full);
    let cat = FullCatalog::read_from(&mut cur, full.len(), true).unwrap();
    assert_eq!(cat.catalog_version, 1);
    assert_eq!(cat.entries.len(), 1);
    let entry = &cat.entries[0];
    assert_eq!(entry.section_type, SectionType::CscShard);
    let stats = entry.stats.as_ref().unwrap();
    // v1 axis-overload reconciled into the column pair.
    assert_eq!(stats.row_start, 100);
    assert_eq!(stats.row_end, 250);
    assert_eq!(stats.col_start, 100);
    assert_eq!(stats.col_end, 250);
    // major_start dispatches on section_type → returns col_start.
    assert_eq!(stats.major_start(SectionType::CscShard), 100);
    assert_eq!(stats.major_end(SectionType::CscShard), 250);
    assert_eq!(stats.col_range(), 100..250);
}

/// `reconcile_v1_csr_col_range(n_vars)` populates `col_end` for
/// row-major shard entries after a v1 catalog read.
#[test]
fn v1_catalog_csr_reconcile_col_range() {
    // Build a v1 catalog with one CSR shard.
    let mut v1_stats_bytes = Vec::new();
    v1_stats_bytes.extend_from_slice(&0u64.to_le_bytes()); // row_start
    v1_stats_bytes.extend_from_slice(&16384u64.to_le_bytes()); // row_end
    v1_stats_bytes.extend_from_slice(&1000u64.to_le_bytes()); // nnz
    v1_stats_bytes.extend_from_slice(&1u32.to_le_bytes()); // value_min
    v1_stats_bytes.extend_from_slice(&255u32.to_le_bytes()); // value_max
    v1_stats_bytes.extend_from_slice(&50000u64.to_le_bytes()); // value_sum
    v1_stats_bytes.push(0);

    let mut payload = Vec::new();
    payload.extend_from_slice(&1u16.to_le_bytes()); // catalog_version=1
    payload.extend_from_slice(&0u64.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    payload.extend_from_slice(&50000u64.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    let name = b"X_shard_0";
    payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
    payload.extend_from_slice(name);
    payload.extend_from_slice(&4352u64.to_le_bytes());
    payload.extend_from_slice(&10000u64.to_le_bytes());
    payload.push(SectionType::CsrShard as u8);
    payload.extend_from_slice(&[0u8; 32]);
    payload.extend_from_slice(&(v1_stats_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(&v1_stats_bytes);

    let checksum = crate::checksum::blake3_hash(&payload);
    let mut full = payload.clone();
    full.extend_from_slice(&checksum);

    let mut cur = Cursor::new(&full);
    let mut cat = FullCatalog::read_from(&mut cur, full.len(), true).unwrap();
    // Before reconcile: col_end == 0 for CSR entries.
    let stats = cat.entries[0].stats.as_ref().unwrap();
    assert_eq!(stats.col_start, 0);
    assert_eq!(stats.col_end, 0);

    cat.reconcile_v1_csr_col_range(30_000);
    let stats = cat.entries[0].stats.as_ref().unwrap();
    assert_eq!(stats.col_start, 0);
    assert_eq!(stats.col_end, 30_000);
}

// -----------------------------------------------------------------------
// LazyShardStats tests
// -----------------------------------------------------------------------

fn stats_with_column_stats() -> ShardStats {
    ShardStats {
        row_start: 0,
        row_end: 256,
        col_start: 0,
        col_end: 5_000,
        nnz: 50_000,
        value_min: 1,
        value_max: 255,
        value_sum: 1_000_000,
        n_indexed_columns: 2,
        column_stats: vec![
            ColumnStat::MinMax {
                column_name_hash: column_name_hash("n_genes"),
                min: 100.0,
                max: 5000.0,
            },
            ColumnStat::CategoryBitset {
                column_name_hash: column_name_hash("cell_type"),
                bitset: vec![0xFF, 0x0F, 0xA5],
            },
        ],
    }
}

/// `n_indexed_columns == 0`: the retained `column_stats_bytes`
/// must be empty (no heap allocation) and `decode_column_stats`
/// must return `Vec::new` without parsing.
#[test]
fn lazy_shard_stats_zero_indexed_columns_empty_tail() {
    let full = sample_stats(); // n_indexed_columns == 0
    let mut buf = Vec::new();
    full.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), SHARD_STATS_BASE_SIZE_V2);

    let lazy =
        LazyShardStats::read_from(&mut Cursor::new(&buf), CURRENT_CATALOG_VERSION, buf.len())
            .unwrap();

    assert_eq!(lazy.n_indexed_columns, 0);
    assert!(
        lazy.column_stats_bytes().is_empty(),
        "n_indexed_columns=0 must retain zero bytes, got {} bytes",
        lazy.column_stats_bytes().len(),
    );

    let decoded = lazy.decode_column_stats().unwrap();
    assert!(decoded.is_empty());

    // to_full() must round-trip cleanly back to the original.
    assert_eq!(lazy.to_full().unwrap(), full);
}

/// `n_indexed_columns > 0`: the retained byte payload is the
/// suffix beyond the fixed-width prefix, and `decode_column_stats`
/// yields exactly the `Vec<ColumnStat>` an eager
/// `ShardStats::read_from` would have produced.
#[test]
fn lazy_shard_stats_with_column_stats_decode_matches_eager() {
    let full = stats_with_column_stats();
    let mut buf = Vec::new();
    full.write_to(&mut buf).unwrap();
    assert!(buf.len() > SHARD_STATS_BASE_SIZE_V2);

    let lazy =
        LazyShardStats::read_from(&mut Cursor::new(&buf), CURRENT_CATALOG_VERSION, buf.len())
            .unwrap();

    assert_eq!(lazy.row_start, full.row_start);
    assert_eq!(lazy.row_end, full.row_end);
    assert_eq!(lazy.col_start, full.col_start);
    assert_eq!(lazy.col_end, full.col_end);
    assert_eq!(lazy.nnz, full.nnz);
    assert_eq!(lazy.value_min, full.value_min);
    assert_eq!(lazy.value_max, full.value_max);
    assert_eq!(lazy.value_sum, full.value_sum);
    assert_eq!(lazy.n_indexed_columns, full.n_indexed_columns);
    assert_eq!(
        lazy.column_stats_bytes().len(),
        buf.len() - SHARD_STATS_BASE_SIZE_V2,
    );

    let decoded = lazy.decode_column_stats().unwrap();
    assert_eq!(decoded, full.column_stats);

    // Re-decoding is idempotent (no internal mutation).
    assert_eq!(lazy.decode_column_stats().unwrap(), full.column_stats);
}

/// `to_full()` produces a struct byte-identical to the eager
/// `ShardStats::read_from` of the same payload.
#[test]
fn lazy_shard_stats_to_full_round_trip() {
    let full = stats_with_column_stats();
    let mut buf = Vec::new();
    full.write_to(&mut buf).unwrap();

    let eager = ShardStats::read_from(&mut Cursor::new(&buf), CURRENT_CATALOG_VERSION).unwrap();
    let lazy =
        LazyShardStats::read_from(&mut Cursor::new(&buf), CURRENT_CATALOG_VERSION, buf.len())
            .unwrap();
    assert_eq!(lazy.to_full().unwrap(), eager);
}

/// `from_full().to_full()` round-trips an in-memory `ShardStats`
/// without touching disk.
#[test]
fn lazy_shard_stats_from_full_round_trip() {
    let full = stats_with_column_stats();
    let lazy = LazyShardStats::from_full(&full, CURRENT_CATALOG_VERSION).unwrap();
    assert_eq!(lazy.to_full().unwrap(), full);

    let empty = sample_stats();
    let lazy_empty = LazyShardStats::from_full(&empty, CURRENT_CATALOG_VERSION).unwrap();
    assert!(lazy_empty.column_stats_bytes().is_empty());
    assert_eq!(lazy_empty.to_full().unwrap(), empty);
}

/// v1 stats: 41-byte prefix, no `col_start`/`col_end`. Lazy parser
/// must accept the legacy layout and leave `col_*` zero. v1 files
/// did not ship `column_stats` in practice (`n_indexed_columns`
/// was always 0 in Phase 1), but the decode pipeline still has to
/// support a future v1 catalog with a non-empty tail — confirm
/// the byte buffer is sized correctly off the `stats_payload_len`
/// argument rather than a hard-coded prefix.
#[test]
fn lazy_shard_stats_v1_layout() {
    // Hand-build a 41-byte v1 stats payload.
    let mut v1 = Vec::new();
    v1.write_u64::<LittleEndian>(0).unwrap();
    v1.write_u64::<LittleEndian>(256).unwrap();
    v1.write_u64::<LittleEndian>(50_000).unwrap(); // nnz
    v1.write_u32::<LittleEndian>(1).unwrap();
    v1.write_u32::<LittleEndian>(255).unwrap();
    v1.write_u64::<LittleEndian>(1_000_000).unwrap();
    v1.push(0); // n_indexed_columns = 0
    assert_eq!(v1.len(), SHARD_STATS_BASE_SIZE_V1);

    let lazy = LazyShardStats::read_from(&mut Cursor::new(&v1), 1, v1.len()).unwrap();
    assert_eq!(lazy.row_start, 0);
    assert_eq!(lazy.row_end, 256);
    assert_eq!(lazy.col_start, 0, "v1 must leave col_start zero");
    assert_eq!(lazy.col_end, 0, "v1 must leave col_end zero");
    assert_eq!(lazy.nnz, 50_000);
    assert_eq!(lazy.value_min, 1);
    assert_eq!(lazy.value_max, 255);
    assert_eq!(lazy.value_sum, 1_000_000);
    assert_eq!(lazy.n_indexed_columns, 0);
    assert!(lazy.column_stats_bytes().is_empty());
    assert_eq!(lazy.catalog_version(), 1);
}

/// A `stats_payload_len` smaller than the fixed-width prefix
/// must produce a clean error rather than reading past the
/// buffer or returning garbage scalars.
#[test]
fn lazy_shard_stats_rejects_undersize_payload() {
    let mut buf = vec![0u8; 8];
    let err = LazyShardStats::read_from(&mut Cursor::new(&buf), CURRENT_CATALOG_VERSION, buf.len())
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("too short") || msg.contains("UnexpectedEof"),
        "expected size-rejection error, got: {msg}"
    );

    // v1 prefix is 41 bytes; passing 40 must also reject.
    buf.resize(40, 0);
    let err = LazyShardStats::read_from(&mut Cursor::new(&buf), 1, buf.len()).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("too short") || msg.contains("UnexpectedEof"));
}

/// Predicate-pushdown analogue: when a query requires column
/// statistics, `decode_column_stats()` produces the same
/// `Vec<ColumnStat>` that `scx-engine::pushdown` would receive
/// from an eagerly-parsed `ShardStats`. This is the test the
/// Phase 4 spec calls out as "predicate pushdown paths that
/// force lazy stats decoding".
#[test]
fn lazy_shard_stats_pushdown_force_decode() {
    // Build a stats payload whose column_stats describe a
    // realistic obs-predicate filter (numeric range + categorical
    // bitset). Then confirm the lazy path returns exactly the
    // structures the eager `ShardStats::column_stats` would expose.
    let full = ShardStats {
        row_start: 16_384,
        row_end: 32_768,
        col_start: 0,
        col_end: 60_000,
        nnz: 5_000_000,
        value_min: 0,
        value_max: 65_535,
        value_sum: 100_000_000,
        n_indexed_columns: 2,
        column_stats: vec![
            ColumnStat::MinMax {
                column_name_hash: column_name_hash("total_counts"),
                min: 500.0,
                max: 50_000.0,
            },
            ColumnStat::CategoryBitset {
                column_name_hash: column_name_hash("cell_type"),
                bitset: vec![0b0000_1011, 0b1100_0000, 0b0000_0001],
            },
        ],
    };
    let mut buf = Vec::new();
    full.write_to(&mut buf).unwrap();

    let lazy =
        LazyShardStats::read_from(&mut Cursor::new(&buf), CURRENT_CATALOG_VERSION, buf.len())
            .unwrap();

    // Cold scalar access: no column-stats decoding has happened.
    assert_eq!(lazy.nnz, 5_000_000);
    assert_eq!(lazy.n_indexed_columns, 2);
    assert!(!lazy.column_stats_bytes().is_empty());

    // Force decode (the predicate-pushdown analogue).
    let forced = lazy.decode_column_stats().unwrap();
    assert_eq!(forced.len(), 2);
    match &forced[0] {
        ColumnStat::MinMax {
            column_name_hash,
            min,
            max,
        } => {
            assert_eq!(*column_name_hash, column_name_hash_of("total_counts"));
            assert_eq!(*min, 500.0);
            assert_eq!(*max, 50_000.0);
        }
        _ => panic!("expected MinMax for total_counts"),
    }
    match &forced[1] {
        ColumnStat::CategoryBitset {
            column_name_hash,
            bitset,
        } => {
            assert_eq!(*column_name_hash, column_name_hash_of("cell_type"));
            assert_eq!(bitset, &vec![0b0000_1011, 0b1100_0000, 0b0000_0001]);
        }
        _ => panic!("expected CategoryBitset for cell_type"),
    }
}

fn column_name_hash_of(name: &str) -> u64 {
    column_name_hash(name)
}

// -----------------------------------------------------------------------
// FullCatalog tests (9.13–9.16)
// -----------------------------------------------------------------------

fn sample_full_entry(name: &str, stype: SectionType, with_stats: bool) -> FullCatalogEntry {
    FullCatalogEntry {
        name: name.to_string(),
        offset: 4352 + (name.len() as u64) * 1000,
        length: 50_000,
        section_type: stype,
        checksum: [0xCD; 32],
        modality_id: 0,
        stats: if with_stats {
            Some(sample_stats())
        } else {
            None
        },
    }
}

fn sample_full_catalog() -> FullCatalog {
    FullCatalog {
        catalog_version: CURRENT_CATALOG_VERSION,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        n_obs: 50_000,
        entries: vec![
            sample_full_entry("obs", SectionType::ObsMetadata, false),
            sample_full_entry("obs_index", SectionType::ObsIndex, false),
            sample_full_entry("var", SectionType::VarMetadata, false),
            sample_full_entry("var_index", SectionType::VarIndex, false),
            sample_full_entry("X_shard_0", SectionType::CsrShard, true),
            sample_full_entry("X_shard_1", SectionType::CsrShard, true),
            sample_full_entry("X_shard_2", SectionType::CsrShard, true),
            sample_full_entry("X_shard_3", SectionType::CsrShard, true),
            sample_full_entry("provenance", SectionType::Provenance, false),
            sample_full_entry("uns", SectionType::UnsBlob, false),
        ],
        data_generation: 0,
        csc_build_generation: 0,
    }
}

/// 9.13: Write FullCatalog with 10 entries → read back → all fields match
#[test]
fn full_catalog_round_trip() {
    let catalog = sample_full_catalog();
    let mut buf = Vec::new();
    catalog.write_to(&mut buf).unwrap();

    let total_len = buf.len();
    let mut cursor = Cursor::new(&buf);
    let decoded = FullCatalog::read_from(&mut cursor, total_len, true).unwrap();

    assert_eq!(decoded.catalog_version, catalog.catalog_version);
    assert_eq!(decoded.manifest_sequence, catalog.manifest_sequence);
    assert_eq!(decoded.prev_catalog_offset, catalog.prev_catalog_offset);
    assert_eq!(decoded.n_obs, catalog.n_obs);
    assert_eq!(decoded.entries.len(), catalog.entries.len());

    for (orig, dec) in catalog.entries.iter().zip(decoded.entries.iter()) {
        assert_eq!(dec.name, orig.name);
        assert_eq!(dec.offset, orig.offset);
        assert_eq!(dec.length, orig.length);
        assert_eq!(dec.section_type, orig.section_type);
        assert_eq!(dec.checksum, orig.checksum);
        assert_eq!(dec.stats.is_some(), orig.stats.is_some());
        if let (Some(os), Some(ds)) = (&orig.stats, &dec.stats) {
            assert_eq!(ds, os);
        }
    }
}

/// 9.14: Corrupt 1 byte → ChecksumMismatch
#[test]
fn full_catalog_checksum_mismatch() {
    let catalog = sample_full_catalog();
    let mut buf = Vec::new();
    catalog.write_to(&mut buf).unwrap();

    // Corrupt a byte in the payload (not in the trailing checksum)
    buf[10] ^= 0xFF;

    let total_len = buf.len();
    let mut cursor = Cursor::new(&buf);
    let err = FullCatalog::read_from(&mut cursor, total_len, true).unwrap_err();
    assert!(matches!(err, ScxError::ChecksumMismatch { .. }));
}

/// 9.15: Empty catalog (0 entries) round-trip
#[test]
fn full_catalog_empty() {
    let catalog = FullCatalog {
        catalog_version: CURRENT_CATALOG_VERSION,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        n_obs: 0,
        entries: vec![],
        data_generation: 0,
        csc_build_generation: 0,
    };

    let mut buf = Vec::new();
    catalog.write_to(&mut buf).unwrap();

    let total_len = buf.len();
    let mut cursor = Cursor::new(&buf);
    let decoded = FullCatalog::read_from(&mut cursor, total_len, true).unwrap();

    assert_eq!(decoded.catalog_version, CURRENT_CATALOG_VERSION);
    assert_eq!(decoded.entries.len(), 0);
}

/// 9.16: Lookup methods
#[test]
fn full_catalog_get_by_name() {
    let catalog = sample_full_catalog();

    let entry = catalog.get("obs").unwrap();
    assert_eq!(entry.section_type, SectionType::ObsMetadata);

    let entry = catalog.get("X_shard_2").unwrap();
    assert_eq!(entry.section_type, SectionType::CsrShard);

    assert!(catalog.get("nonexistent").is_none());
}

#[test]
fn full_catalog_shards_by_type() {
    let catalog = sample_full_catalog();

    let csr_shards = catalog.shards(SectionType::CsrShard);
    assert_eq!(csr_shards.len(), 4);
    for s in &csr_shards {
        assert_eq!(s.section_type, SectionType::CsrShard);
    }

    let obs = catalog.shards(SectionType::ObsMetadata);
    assert_eq!(obs.len(), 1);

    let empty = catalog.shards(SectionType::CscShard);
    assert_eq!(empty.len(), 0);
}

#[test]
fn full_catalog_shards_sorted() {
    let mut catalog = sample_full_catalog();
    // Set distinct row_start values in reverse order for the 4 CSR shards
    let mut shard_idx = 0u64;
    for entry in catalog.entries.iter_mut() {
        if let Some(ref mut stats) = entry.stats {
            stats.row_start = (3 - shard_idx) * 16384;
            shard_idx += 1;
        }
    }

    let sorted = catalog.shards_sorted();
    assert_eq!(sorted.len(), 4);
    let starts: Vec<u64> = sorted
        .iter()
        .map(|e| e.stats.as_ref().unwrap().row_start)
        .collect();
    assert!(starts.windows(2).all(|w| w[0] <= w[1]));
}

/// 4-shard CSC layout with non-uniform column sizes.
/// Confirm csc_shards_for_col_range returns exactly the overlapping
/// subset for various queries, and that it sorts the result by
/// `major_start()`.
#[test]
fn csc_shards_for_col_range_filters_overlapping() {
    // 4 CSC shards covering [0, 100), [100, 250), [250, 260), [260, 1000).
    let cscs: Vec<(u64, u64)> = vec![(0, 100), (100, 250), (250, 260), (260, 1000)];

    let mut entries = Vec::new();
    // Insert in shuffled order so we exercise the sort.
    let order = [2usize, 0, 3, 1];
    for &i in &order {
        let (lo, hi) = cscs[i];
        entries.push(FullCatalogEntry {
            name: format!("X_csc_shard_{i}"),
            offset: 4352 + (i as u64) * 1_000_000,
            length: 50_000,
            section_type: SectionType::CscShard,
            checksum: [0u8; 32],
            modality_id: 0,
            stats: Some(ShardStats {
                row_start: 0, // v2 CSC: row pair carries [0, n_obs)
                row_end: 1000,
                col_start: lo,
                col_end: hi,
                nnz: 1000,
                value_min: 0,
                value_max: 0,
                value_sum: 0,
                n_indexed_columns: 0,
                column_stats: vec![],
            }),
        });
    }
    // Sprinkle in a CSR shard that should never be selected.
    entries.push(FullCatalogEntry {
        name: "X_shard_0".to_string(),
        offset: 0,
        length: 1,
        section_type: SectionType::CsrShard,
        checksum: [0u8; 32],
        modality_id: 0,
        stats: Some(ShardStats {
            row_start: 0,
            row_end: 100,
            col_start: 0,
            col_end: 1000,
            nnz: 0,
            value_min: 0,
            value_max: 0,
            value_sum: 0,
            n_indexed_columns: 0,
            column_stats: vec![],
        }),
    });

    let catalog = FullCatalog {
        catalog_version: CURRENT_CATALOG_VERSION,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        n_obs: 1000,
        entries,
        data_generation: 0,
        csc_build_generation: 0,
    };

    // Whole range: all 4 CSC shards in sorted order.
    let all = catalog.csc_shards_for_col_range(0, 1000);
    let starts: Vec<u64> = all
        .iter()
        .map(|e| e.stats.as_ref().unwrap().major_start(SectionType::CscShard))
        .collect();
    assert_eq!(starts, vec![0, 100, 250, 260]);

    // Partial overlap on shards 0 and 1.
    let mid = catalog.csc_shards_for_col_range(50, 200);
    let starts: Vec<u64> = mid
        .iter()
        .map(|e| e.stats.as_ref().unwrap().major_start(SectionType::CscShard))
        .collect();
    assert_eq!(starts, vec![0, 100]);

    // Hits only the tiny shard 2 (covers [250, 260)) and shard 3.
    let tiny = catalog.csc_shards_for_col_range(255, 300);
    let starts: Vec<u64> = tiny
        .iter()
        .map(|e| e.stats.as_ref().unwrap().major_start(SectionType::CscShard))
        .collect();
    assert_eq!(starts, vec![250, 260]);

    // Exact-boundary query: [100, 250) is shard 1 alone (right edge
    // is exclusive, so shard 2 [250, 260) is NOT included).
    let exact = catalog.csc_shards_for_col_range(100, 250);
    let starts: Vec<u64> = exact
        .iter()
        .map(|e| e.stats.as_ref().unwrap().major_start(SectionType::CscShard))
        .collect();
    assert_eq!(starts, vec![100]);

    // Past the end: empty.
    let past = catalog.csc_shards_for_col_range(2000, 3000);
    assert!(past.is_empty());

    // Empty range (lo == hi or lo > hi): empty.
    assert!(catalog.csc_shards_for_col_range(50, 50).is_empty());
    assert!(catalog.csc_shards_for_col_range(200, 100).is_empty());
}

/// Regression: 2026-05-10 tier-full gate run #2.
///
/// Before the fix, `FullCatalog::write_to` preserved `catalog_version`
/// from `self` while `ShardStats::write_to` always emitted the v2 stats
/// layout (with explicit `col_start` / `col_end`). When a v1 catalog
/// (e.g. one parsed from an older `.scx` file by `scx-cloud::push`) was
/// re-serialised, the on-disk header claimed v1 but the per-entry stats
/// were v2-shaped. Readers branched on the header, parsed v1-sized
/// stats, and walked off the end of the buffer — surfacing as
/// `failed to fill whole buffer` on every `pyscx.pull`.
///
/// Fix: `write_to` upgrades `catalog_version` to at least 2 before
/// serialising. v1 catalogs round-trip into v2 catalogs that any
/// reader can parse cleanly.
#[test]
fn write_upgrades_v1_catalog_to_v2() {
    let v1 = FullCatalog {
        catalog_version: 1,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        n_obs: 1_000,
        entries: vec![
            sample_full_entry("obs", SectionType::ObsMetadata, false),
            sample_full_entry("X_shard_0", SectionType::CsrShard, true),
        ],
        data_generation: 0,
        csc_build_generation: 0,
    };

    let mut buf = Vec::new();
    v1.write_to(&mut buf).unwrap();

    // The first 2 bytes of the catalog payload are the catalog_version
    // (u16 LE). Auto-upgrade should land 2 on disk even though the
    // source struct said 1.
    let on_disk_version = u16::from_le_bytes([buf[0], buf[1]]);
    assert_eq!(
        on_disk_version, 2,
        "expected catalog_version=2 on disk (v1 should auto-upgrade)"
    );

    // Round-trip parses cleanly — pre-fix, this raised
    // `failed to fill whole buffer` because v1 stats parsing
    // misaligned against the v2-shaped bytes.
    let total_len = buf.len();
    let parsed = FullCatalog::read_from(&mut std::io::Cursor::new(&buf), total_len, true).unwrap();
    assert_eq!(parsed.catalog_version, 2);
    assert_eq!(parsed.n_obs, v1.n_obs);
    assert_eq!(parsed.entries.len(), v1.entries.len());
}

/// v4 generation counters round-trip when non-zero, and the on-disk
/// catalog declares v4 so the trailing fields are read back.
#[test]
fn catalog_v4_generation_round_trip() {
    let mut cat = sample_full_catalog();
    cat.catalog_version = 2; // start below v4; the counters force the upgrade
    cat.data_generation = 7;
    cat.csc_build_generation = 5;

    let mut buf = Vec::new();
    cat.write_to(&mut buf).unwrap();

    // A non-zero counter upgrades the declared version to 4 on disk.
    let on_disk_version = u16::from_le_bytes([buf[0], buf[1]]);
    assert_eq!(on_disk_version, 4, "non-zero generation must stamp v4");

    let total_len = buf.len();
    let parsed = FullCatalog::read_from(&mut Cursor::new(&buf), total_len, true).unwrap();
    assert_eq!(parsed.catalog_version, 4);
    assert_eq!(parsed.data_generation, 7);
    assert_eq!(parsed.csc_build_generation, 5);
    // Checksum still validates with the trailing fields inside the payload.
}

/// Zero counters emit no trailing bytes and keep the v2 declared
/// version — existing zero-counter files stay byte-identical, and a
/// reader defaults both counters to 0.
#[test]
fn catalog_zero_generation_stays_v2_no_trailing() {
    let mut with_zero = sample_full_catalog();
    with_zero.catalog_version = 2;
    with_zero.data_generation = 0;
    with_zero.csc_build_generation = 0;
    let mut buf_zero = Vec::new();
    with_zero.write_to(&mut buf_zero).unwrap();
    assert_eq!(
        u16::from_le_bytes([buf_zero[0], buf_zero[1]]),
        2,
        "zero counters must not bump the declared version"
    );

    // Compare against the same catalog re-serialised: byte length must
    // not grow by the 16 trailing bytes (none are written).
    let parsed = FullCatalog::read_from(&mut Cursor::new(&buf_zero), buf_zero.len(), true).unwrap();
    assert_eq!(parsed.data_generation, 0);
    assert_eq!(parsed.csc_build_generation, 0);
    assert_eq!(parsed.catalog_version, 2);
}

/// A v2/v3-shaped catalog (no trailing generation bytes) reads back
/// with both counters defaulted to 0 — the legacy-file path.
#[test]
fn catalog_pre_v4_defaults_generation_to_zero() {
    let mut cat = sample_full_catalog();
    cat.catalog_version = 2;
    // counters left at 0 → no trailing bytes written.
    let mut buf = Vec::new();
    cat.write_to(&mut buf).unwrap();

    let parsed = FullCatalog::read_from(&mut Cursor::new(&buf), buf.len(), true).unwrap();
    assert_eq!(parsed.data_generation, 0);
    assert_eq!(parsed.csc_build_generation, 0);
}

// -----------------------------------------------------------------------
// Defensive allocation cap tests (Patch 9)
// -----------------------------------------------------------------------

#[test]
fn catalog_rejects_oversized_n_entries() {
    // Craft a minimal catalog payload with n_entries = u32::MAX.
    // The catalog preamble is:
    //   version(u16) + manifest_seq(u64) + prev_offset(u64) +
    //   n_obs(u64) + n_entries(u32) = 30 bytes
    // Plus 32 bytes trailing checksum = 62 bytes minimum.
    let mut payload = Vec::new();
    use byteorder::WriteBytesExt;
    payload.write_u16::<LittleEndian>(2).unwrap(); // catalog_version
    payload.write_u64::<LittleEndian>(1).unwrap(); // manifest_sequence
    payload.write_u64::<LittleEndian>(0).unwrap(); // prev_catalog_offset
    payload.write_u64::<LittleEndian>(100).unwrap(); // n_obs
    payload.write_u32::<LittleEndian>(u32::MAX).unwrap(); // n_entries = absurd

    // Append a dummy 32-byte checksum (verify_checksum = false).
    payload.extend_from_slice(&[0u8; 32]);

    let total_len = payload.len();
    let result = FullCatalog::read_from(&mut std::io::Cursor::new(&payload), total_len, false);
    assert!(result.is_err(), "should reject oversized n_entries");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("allocation too large"),
        "error should mention allocation: {err_msg}"
    );
}

// -----------------------------------------------------------------------
// Phase 2 parser: malformed-input regression tests
// -----------------------------------------------------------------------

/// `stats_len` claims more bytes than remain in the payload. The
/// borrowed-slice parser must raise `UnexpectedEof` rather than
/// reading past the buffer or silently truncating the next entry.
#[test]
fn catalog_rejects_truncated_stats_payload() {
    // v2 header for a single CSR entry whose stats_len lies.
    let mut payload = Vec::new();
    payload.write_u16::<LittleEndian>(2).unwrap(); // catalog_version
    payload.write_u64::<LittleEndian>(0).unwrap(); // manifest_sequence
    payload.write_u64::<LittleEndian>(0).unwrap(); // prev_catalog_offset
    payload.write_u64::<LittleEndian>(100).unwrap(); // n_obs
    payload.write_u32::<LittleEndian>(1).unwrap(); // n_entries = 1

    let name = b"X_shard_0";
    payload
        .write_u16::<LittleEndian>(name.len() as u16)
        .unwrap();
    payload.extend_from_slice(name);
    payload.write_u64::<LittleEndian>(4352).unwrap(); // offset
    payload.write_u64::<LittleEndian>(1000).unwrap(); // length
    payload.push(SectionType::CsrShard as u8);
    payload.extend_from_slice(&[0u8; 32]); // checksum
    payload.push(0); // modality_id (v2)
                     // Claim 200 bytes of stats but only write 8 (the row_start u64).
    payload.write_u16::<LittleEndian>(200).unwrap();
    payload.extend_from_slice(&0u64.to_le_bytes());

    payload.extend_from_slice(&[0u8; 32]); // trailing checksum slot
    let total_len = payload.len();
    let err =
        FullCatalog::read_from(&mut std::io::Cursor::new(&payload), total_len, false).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("stats payload truncated") || msg.contains("UnexpectedEof"),
        "expected stats-truncated error, got: {msg}"
    );
}

/// Invalid UTF-8 in a catalog entry's `name` field must surface as
/// an `InvalidData` IO error. The borrowed-slice parser validates
/// in place against the payload slice; we must still produce the
/// same diagnostic the old `String::from_utf8` path produced.
#[test]
fn catalog_rejects_invalid_utf8_name() {
    let mut payload = Vec::new();
    payload.write_u16::<LittleEndian>(2).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u32::<LittleEndian>(1).unwrap();

    // Lone continuation byte 0x80 — invalid UTF-8 start byte.
    let bad_name: &[u8] = &[0x80, 0x80, 0x80];
    payload
        .write_u16::<LittleEndian>(bad_name.len() as u16)
        .unwrap();
    payload.extend_from_slice(bad_name);
    payload.write_u64::<LittleEndian>(4352).unwrap();
    payload.write_u64::<LittleEndian>(1000).unwrap();
    payload.push(SectionType::ObsMetadata as u8);
    payload.extend_from_slice(&[0u8; 32]);
    payload.push(0); // modality_id (v2)
    payload.write_u16::<LittleEndian>(0).unwrap(); // no stats

    payload.extend_from_slice(&[0u8; 32]);
    let total_len = payload.len();
    let err =
        FullCatalog::read_from(&mut std::io::Cursor::new(&payload), total_len, false).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("invalid utf-8") || msg.contains("InvalidData") || msg.contains("utf-8"),
        "expected utf-8 error, got: {msg}"
    );
}

/// `name_len` exceeds the bytes remaining in the payload. Mirror
/// of the stats-truncation case but on the name field — exercises
/// the per-entry `split_at_checked` rather than the upfront
/// `validate_allocation` cap. The payload is padded so that
/// `n_entries * MIN_ENTRY_BYTES <= payload_len` (the cap accepts
/// it); only the in-loop check catches the lie.
#[test]
fn catalog_rejects_truncated_entry_name() {
    let mut payload = Vec::new();
    payload.write_u16::<LittleEndian>(2).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u32::<LittleEndian>(1).unwrap();

    // Claim 1024 bytes of name but only write 4 ("abcd"). The
    // parser must reject rather than reading garbage out of the
    // following fixed-width fields.
    payload.write_u16::<LittleEndian>(1024).unwrap();
    payload.extend_from_slice(b"abcd");
    // Pad past `MIN_ENTRY_BYTES = 53` so the upfront allocation
    // cap doesn't short-circuit the test before the entry loop
    // runs.
    payload.extend_from_slice(&[0u8; 100]);

    payload.extend_from_slice(&[0u8; 32]); // trailing checksum slot
    let total_len = payload.len();
    let err =
        FullCatalog::read_from(&mut std::io::Cursor::new(&payload), total_len, false).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("name truncated") || msg.contains("UnexpectedEof"),
        "expected name-truncated error, got: {msg}"
    );
}
