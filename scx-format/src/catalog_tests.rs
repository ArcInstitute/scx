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
fn root_catalog_read_rejects_oversized_count() {
    // SCX-007: a corrupt entry count larger than the root region can hold must
    // be rejected before allocating/reading, not trusted.
    use byteorder::{LittleEndian, WriteBytesExt};
    let max_entries = ROOT_CATALOG_MAX_SIZE / ROOT_CATALOG_ENTRY_SIZE; // 77
    let mut buf = Vec::new();
    // Declare far more entries than the region could contain.
    buf.write_u16::<LittleEndian>((max_entries as u16) + 1)
        .unwrap();
    let mut cursor = std::io::Cursor::new(buf);
    let err = RootCatalog::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, ScxError::InvalidCatalog(_)));

    // A legal count still round-trips.
    let catalog = RootCatalog {
        n_section_groups: 2,
        entries: vec![sample_root_entry(0), sample_root_entry(1)],
    };
    let mut ok_buf = Vec::new();
    catalog.write_to(&mut ok_buf).unwrap();
    let decoded = RootCatalog::read_from(&mut std::io::Cursor::new(ok_buf)).unwrap();
    assert_eq!(decoded.entries.len(), 2);
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

#[test]
fn dense_mapping_shards_sorted_filters_and_sorts() {
    // Three obsm shards for key "X_pca" (out of order), plus a shard for
    // a different key and a different modality that must be excluded.
    let mk = |name: &str, modality_id: u8, row_start: u64| {
        let mut s = sample_stats();
        s.row_start = row_start;
        FullCatalogEntry {
            name: name.to_string(),
            offset: 4352,
            length: 50_000,
            section_type: SectionType::ObsmEmbeddingShard,
            checksum: [0u8; 32],
            modality_id,
            stats: Some(s),
        }
    };
    let catalog = FullCatalog {
        catalog_version: CURRENT_CATALOG_VERSION,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        n_obs: 100,
        entries: vec![
            mk("obsm/X_pca_shard_2", 0, 200),
            mk("obsm/X_pca_shard_0", 0, 0),
            mk("obsm/X_pca_shard_1", 0, 100),
            mk("obsm/X_umap_shard_0", 0, 0), // different key
            mk("obsm/X_pca_shard_0", 1, 0),  // different modality
        ],
        data_generation: 0,
        csc_build_generation: 0,
    };

    let sorted = catalog.dense_mapping_shards_sorted(
        SectionType::ObsmEmbeddingShard,
        0,
        "obsm/X_pca_shard_",
    );
    let names: Vec<&str> = sorted.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "obsm/X_pca_shard_0",
            "obsm/X_pca_shard_1",
            "obsm/X_pca_shard_2"
        ],
        "only modality-0 X_pca shards, sorted by row_start"
    );
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

#[test]
fn list_logical_names_dedups_legacy_and_sharded() {
    let mut cat = sample_full_catalog();
    cat.entries.extend([
        // Legacy single-section obsm.
        sample_full_entry("obsm/X_pca", SectionType::ObsmEmbedding, false),
        // Sharded obsm across two shards → one logical name.
        sample_full_entry(
            "obsm/X_umap_shard_0",
            SectionType::ObsmEmbeddingShard,
            false,
        ),
        sample_full_entry(
            "obsm/X_umap_shard_1",
            SectionType::ObsmEmbeddingShard,
            false,
        ),
        // Sharded varm.
        sample_full_entry("varm/PCs_shard_0", SectionType::VarmEmbeddingShard, false),
        // Two layer shards → one logical layer.
        sample_full_entry("counts_shard_0", SectionType::LayerCsrShard, true),
        sample_full_entry("counts_shard_1", SectionType::LayerCsrShard, true),
    ]);

    let obsm = cat.list_logical_names(
        "obsm",
        SectionType::ObsmEmbedding,
        SectionType::ObsmEmbeddingShard,
    );
    assert_eq!(obsm, vec!["X_pca".to_string(), "X_umap".to_string()]);

    let varm = cat.list_logical_names(
        "varm",
        SectionType::VarmEmbedding,
        SectionType::VarmEmbeddingShard,
    );
    assert_eq!(varm, vec!["PCs".to_string()]);

    assert_eq!(cat.layer_names(), vec!["counts".to_string()]);
}

#[test]
fn list_logical_names_empty_when_absent() {
    // sample_full_catalog has no obsm/varm/layer entries.
    let cat = sample_full_catalog();
    assert!(cat
        .list_logical_names(
            "obsm",
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard
        )
        .is_empty());
    assert!(cat.layer_names().is_empty());
}

// -----------------------------------------------------------------------
// Multimodal query-pushdown prerequisites.
// The engine's modality-scoped scan replaces `csr_shards_sorted()`
// with `csr_shards_for_modality(modality_id)`; on a single-modality/v1 file
// the two MUST be order-and-set identical so the default path stays
// byte-for-byte unchanged.
// -----------------------------------------------------------------------

/// A CSR shard entry for `modality_id` covering `[row_start, row_end)`.
fn csr_shard_entry(name: &str, modality_id: u8, row_start: u64, row_end: u64) -> FullCatalogEntry {
    let mut s = sample_stats();
    s.row_start = row_start;
    s.row_end = row_end;
    FullCatalogEntry {
        name: name.to_string(),
        offset: 4352,
        length: 50_000,
        section_type: SectionType::CsrShard,
        checksum: [0u8; 32],
        modality_id,
        stats: Some(s),
    }
}

#[test]
fn csr_shards_for_modality_0_matches_shards_sorted_single_modality() {
    // sample_full_catalog is a single-modality file (all entries modality 0)
    // with four CSR shards. The modality-0 scan must equal the flattened scan
    // used by the current engine, in the same order.
    let cat = sample_full_catalog();
    let flattened: Vec<&str> = cat
        .csr_shards_sorted()
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    let mod0: Vec<&str> = cat
        .csr_shards_for_modality(0)
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(
        flattened, mod0,
        "single-modality csr_shards_for_modality(0) must equal csr_shards_sorted()"
    );
    // shards_sorted() is the alias the engine actually calls today.
    let alias: Vec<&str> = cat
        .shards_sorted()
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(alias, mod0);
}

#[test]
fn modality_csr_ranges_tile_obs_positive_and_negative() {
    // Two modalities, each independently tiling [0, 300): the multimodal
    // invariant the query pushdown relies on. Insert shuffled so the sort in
    // csr_shards_for_modality is exercised.
    let good = FullCatalog {
        catalog_version: CURRENT_CATALOG_VERSION,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        n_obs: 300,
        entries: vec![
            csr_shard_entry("X/rna/shard_1", 1, 100, 300),
            csr_shard_entry("X/rna/shard_0", 1, 0, 100),
            csr_shard_entry("X/adt/shard_0", 2, 0, 150),
            csr_shard_entry("X/adt/shard_1", 2, 150, 300),
        ],
        data_generation: 0,
        csc_build_generation: 0,
    };
    assert!(
        good.modality_csr_ranges_tile_obs(1, 300),
        "rna tiles [0,300)"
    );
    assert!(
        good.modality_csr_ranges_tile_obs(2, 300),
        "adt tiles [0,300)"
    );
    // Wrong n_obs → not covering.
    assert!(!good.modality_csr_ranges_tile_obs(1, 301));

    // Gap: shard covers [0,100) then [150,300) → 100..150 missing.
    let gap = FullCatalog {
        n_obs: 300,
        entries: vec![
            csr_shard_entry("X/rna/shard_0", 1, 0, 100),
            csr_shard_entry("X/rna/shard_1", 1, 150, 300),
        ],
        ..good.clone()
    };
    assert!(
        !gap.modality_csr_ranges_tile_obs(1, 300),
        "gap breaks tiling"
    );

    // Overlap: [0,200) and [100,300) overlap at 100..200.
    let overlap = FullCatalog {
        n_obs: 300,
        entries: vec![
            csr_shard_entry("X/rna/shard_0", 1, 0, 200),
            csr_shard_entry("X/rna/shard_1", 1, 100, 300),
        ],
        ..good.clone()
    };
    assert!(
        !overlap.modality_csr_ranges_tile_obs(1, 300),
        "overlap breaks tiling"
    );

    // Missing stats → not covering.
    let mut no_stats = csr_shard_entry("X/rna/shard_0", 1, 0, 300);
    no_stats.stats = None;
    let unstatted = FullCatalog {
        n_obs: 300,
        entries: vec![no_stats],
        ..good.clone()
    };
    assert!(!unstatted.modality_csr_ranges_tile_obs(1, 300));

    // Empty shard list covers only n_obs == 0.
    assert!(good.modality_csr_ranges_tile_obs(7, 0));
    assert!(!good.modality_csr_ranges_tile_obs(7, 300));
}

// -----------------------------------------------------------------------
// Forward compatibility: unknown section types (§4.2)
// -----------------------------------------------------------------------

/// A section type no reader knows. Asserted unassigned by every test that
/// uses it, so this fixture fails loudly the day id 30 is allocated rather
/// than quietly becoming a known-type test.
const UNKNOWN_SECTION_TYPE: u8 = 30;

/// Hand-build a catalog payload plus its trailing BLAKE3 checksum,
/// bypassing [`FullCatalog::write_to`].
///
/// The writer cannot express what these fixtures need: an entry carrying a
/// section type it has no variant for, or a stats blob that is not the
/// current layout. Entries are `(name_bytes, section_type_raw, stats_bytes)`;
/// the two generation counters are emitted only when `catalog_version >= 4`,
/// matching `write_to`.
fn build_raw_catalog(
    catalog_version: u16,
    n_obs: u64,
    entries: &[(&[u8], u8, &[u8])],
    generations: (u64, u64),
) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.write_u16::<LittleEndian>(catalog_version).unwrap();
    payload.write_u64::<LittleEndian>(1).unwrap(); // manifest_sequence
    payload.write_u64::<LittleEndian>(0).unwrap(); // prev_catalog_offset
    payload.write_u64::<LittleEndian>(n_obs).unwrap();
    payload
        .write_u32::<LittleEndian>(entries.len() as u32)
        .unwrap();

    for (i, (name, section_type_raw, stats)) in entries.iter().enumerate() {
        payload
            .write_u16::<LittleEndian>(name.len() as u16)
            .unwrap();
        payload.extend_from_slice(name);
        payload
            .write_u64::<LittleEndian>(4352 + i as u64 * 1000)
            .unwrap(); // offset
        payload.write_u64::<LittleEndian>(1000).unwrap(); // length
        payload.push(*section_type_raw);
        payload.extend_from_slice(&[0u8; 32]); // per-entry checksum
        if catalog_version >= 2 {
            payload.push(0); // modality_id
        }
        payload
            .write_u16::<LittleEndian>(stats.len() as u16)
            .unwrap();
        payload.extend_from_slice(stats);
    }

    if catalog_version >= 4 {
        payload.write_u64::<LittleEndian>(generations.0).unwrap();
        payload.write_u64::<LittleEndian>(generations.1).unwrap();
    }

    let checksum = blake3_hash(&payload);
    let mut out = payload;
    out.extend_from_slice(&checksum);
    out
}

/// A stats blob too short for the current layout: 8 bytes of `row_start`,
/// 8 of `row_end`, and then it stops 4 bytes into `col_start`.
///
/// The length is the whole point of the fixture. The outer entry walk
/// advances by `stats_len` whatever it holds, so a well-formed 57-byte blob
/// — or an over-long one — parses cleanly on a *future* section type and
/// demonstrates nothing. Only a blob shorter than the fixed-width prefix
/// makes the stats decoder run off the end.
fn short_stats_blob() -> Vec<u8> {
    vec![0u8; SHORT_STATS_LEN]
}

const SHORT_STATS_LEN: usize = 20;

/// The fixture is only a §4.2 repro while it is shorter than the layout a
/// stats decoder expects. Checked at compile time so a future layout change
/// cannot quietly turn these tests into ones that pass either way.
const _: () = assert!(SHORT_STATS_LEN < SHARD_STATS_BASE_SIZE_V2);

/// §4.2: a section type this reader does not know, carrying a stats blob
/// shorter than today's layout, must cost that one entry — not the file.
///
/// `FullCatalog::read_from` decodes the stats before it resolves the
/// section type, so the short blob aborts the entire catalog parse and the
/// file will not open at all. The catalog is the index to everything, so a
/// single future section makes every section unreachable.
#[test]
fn unknown_section_type_with_short_stats_blob_still_opens() {
    assert!(
        SectionType::from_u8(UNKNOWN_SECTION_TYPE).is_none(),
        "fixture needs an unassigned section type; id {UNKNOWN_SECTION_TYPE} is now known"
    );

    let stats = short_stats_blob();
    let bytes = build_raw_catalog(
        4,
        100,
        &[
            (b"obs", SectionType::ObsMetadata as u8, &[][..]),
            (b"future_section", UNKNOWN_SECTION_TYPE, &stats[..]),
            (b"var", SectionType::VarMetadata as u8, &[][..]),
        ],
        (0, 0),
    );

    let full = FullCatalog::read_from(&mut Cursor::new(&bytes), bytes.len(), true)
        .expect("an unknown section type must cost its own entry, not the whole catalog");
    let names: Vec<&str> = full.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["obs", "var"], "unknown entry must be dropped");

    // The lightweight parser already resolves the type before touching the
    // stats. Pin it so unifying the two cannot regress it in that direction.
    let view = crate::catalog_view::CatalogView::read_from_bytes(&bytes, true)
        .expect("CatalogView already skips before decoding stats");
    let view_names: Vec<Option<&str>> = view.entries.iter().map(|e| e.name()).collect();
    assert_eq!(view_names, vec![Some("obs"), Some("var")]);
}

/// The mirror of the fixture above, and the reason it is not enough on its
/// own: skipping *before* decoding must not turn into skipping *instead of*
/// decoding. A truncated stats blob on a section type the reader does know
/// is corruption, and both parsers must still refuse it.
#[test]
fn truncated_stats_on_a_known_type_is_still_an_error() {
    let stats = short_stats_blob();
    let bytes = build_raw_catalog(
        4,
        100,
        &[
            (b"obs", SectionType::ObsMetadata as u8, &[][..]),
            (b"X_shard_0", SectionType::CsrShard as u8, &stats[..]),
        ],
        (0, 0),
    );

    let err = FullCatalog::read_from(&mut Cursor::new(&bytes), bytes.len(), true)
        .expect_err("a truncated stats blob on a known type is corruption");
    let msg = format!("{err}");
    assert!(
        msg.contains("failed to fill whole buffer") || msg.contains("UnexpectedEof"),
        "expected a truncated-read error, got: {msg}"
    );

    let view_err = crate::catalog_view::CatalogView::read_from_bytes(&bytes, true)
        .expect_err("the lightweight parser must refuse it too");
    let view_msg = format!("{view_err}");
    assert!(
        view_msg.contains("stats payload too short"),
        "expected a short-stats error, got: {view_msg}"
    );
}

/// Unknown entries are dropped, and dropping them must not disturb the
/// order of the survivors. Nothing pins this today in either direction, so
/// a parser unification could silently reverse or reorder the list — which
/// would be invisible to every other test, since the read paths address
/// shards by `(section_type, major_start)` and would still find them.
#[test]
fn unknown_section_ordering_matches_across_parsers() {
    assert!(SectionType::from_u8(UNKNOWN_SECTION_TYPE).is_none());
    let stats = short_stats_blob();

    let bytes = build_raw_catalog(
        4,
        100,
        &[
            (b"obs", SectionType::ObsMetadata as u8, &[][..]),
            (b"future_a", UNKNOWN_SECTION_TYPE, &stats[..]),
            (b"var", SectionType::VarMetadata as u8, &[][..]),
            (b"future_b", 200, &[][..]),
            (b"uns", SectionType::UnsBlob as u8, &[][..]),
        ],
        (0, 0),
    );

    let full = FullCatalog::read_from(&mut Cursor::new(&bytes), bytes.len(), true).unwrap();
    let full_names: Vec<&str> = full.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(full_names, vec!["obs", "var", "uns"]);

    let view = crate::catalog_view::CatalogView::read_from_bytes(&bytes, true).unwrap();
    let view_names: Vec<Option<&str>> = view.entries.iter().map(|e| e.name()).collect();
    assert_eq!(
        view_names,
        full_names.iter().map(|n| Some(*n)).collect::<Vec<_>>(),
        "both parsers must drop the same entries and keep the same order"
    );
}

/// A name is only worth validating if it is going to be kept. An entry the
/// reader is about to drop for having an unknown section type should not be
/// able to fail the whole catalog on its name encoding.
///
/// The complementary case — invalid UTF-8 on an entry the reader *keeps* —
/// is `catalog_rejects_invalid_utf8_name` above, and must stay an error.
#[test]
fn non_utf8_name_on_a_dropped_entry_is_tolerated() {
    assert!(SectionType::from_u8(UNKNOWN_SECTION_TYPE).is_none());

    // Lone continuation bytes — not a valid UTF-8 sequence.
    let bad_name: &[u8] = &[0x80, 0x80, 0x80];
    let bytes = build_raw_catalog(
        4,
        100,
        &[
            (b"obs", SectionType::ObsMetadata as u8, &[][..]),
            (bad_name, UNKNOWN_SECTION_TYPE, &[][..]),
        ],
        (0, 0),
    );

    let full = FullCatalog::read_from(&mut Cursor::new(&bytes), bytes.len(), true)
        .expect("a dropped entry's name encoding must not fail the catalog");
    let names: Vec<&str> = full.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["obs"]);

    crate::catalog_view::CatalogView::read_from_bytes(&bytes, true)
        .expect("the lightweight parser already tolerates it");
}

// -----------------------------------------------------------------------
// value_max folds (csr_max_value / raw_csr_max_value / layer_csr_max_value)
// -----------------------------------------------------------------------

/// A catalog entry with just the fields the value_max folds look at.
fn max_value_entry(
    name: &str,
    stype: SectionType,
    modality_id: u8,
    value_max: Option<u32>,
) -> FullCatalogEntry {
    FullCatalogEntry {
        modality_id,
        stats: value_max.map(|m| ShardStats {
            value_max: m,
            ..sample_stats()
        }),
        ..sample_full_entry(name, stype, false)
    }
}

fn catalog_with(entries: Vec<FullCatalogEntry>) -> FullCatalog {
    FullCatalog {
        entries,
        ..sample_full_catalog()
    }
}

#[test]
fn csr_max_value_scopes_by_modality() {
    let cat = catalog_with(vec![
        max_value_entry("rna/X_shard_0", SectionType::CsrShard, 0, Some(10)),
        max_value_entry("rna/X_shard_1", SectionType::CsrShard, 0, Some(4)),
        max_value_entry("atac/X_shard_0", SectionType::CsrShard, 1, Some(99)),
    ]);
    assert_eq!(cat.csr_max_value(None), 99);
    assert_eq!(cat.csr_max_value(Some(0)), 10);
    assert_eq!(cat.csr_max_value(Some(1)), 99);
    assert_eq!(cat.csr_max_value(Some(2)), 0);
}

/// A larger `value_max` on any other section family must not leak into the
/// X fold, and the raw fold must see only `RawCsrShard`.
#[test]
fn value_max_folds_are_section_family_isolated() {
    let cat = catalog_with(vec![
        max_value_entry("X_shard_0", SectionType::CsrShard, 0, Some(10)),
        max_value_entry("raw/X_shard_0", SectionType::RawCsrShard, 0, Some(500)),
        max_value_entry(
            "layer/rna/counts/shard_0",
            SectionType::LayerCsrShard,
            0,
            Some(700),
        ),
        max_value_entry("csc_shard_0", SectionType::CscShard, 0, Some(900)),
    ]);
    assert_eq!(cat.csr_max_value(None), 10);
    assert_eq!(cat.raw_csr_max_value(), 500);
    // The layer fold must not leak the larger CSC entry (900). Smaller-valued
    // leaks are pinned elsewhere: an X entry larger than every layer exists in
    // `layer_csr_max_value_scopes_by_modality_and_name` (999 vs 80).
    assert_eq!(cat.layer_csr_max_value(0, None), 700);
}

#[test]
fn layer_csr_max_value_scopes_by_modality_and_name() {
    let cat = catalog_with(vec![
        max_value_entry(
            "layer/rna/counts/shard_0",
            SectionType::LayerCsrShard,
            0,
            Some(50),
        ),
        // Trap: contains "counts" as a substring but is a different layer.
        max_value_entry(
            "layer/rna/counts_sq/shard_0",
            SectionType::LayerCsrShard,
            0,
            Some(80),
        ),
        max_value_entry(
            "layer/atac/counts/shard_0",
            SectionType::LayerCsrShard,
            1,
            Some(200),
        ),
        max_value_entry("X_shard_0", SectionType::CsrShard, 0, Some(999)),
    ]);
    // None = every layer of the modality; X shards never contribute.
    assert_eq!(cat.layer_csr_max_value(0, None), 80);
    assert_eq!(cat.layer_csr_max_value(1, None), 200);
    assert_eq!(cat.layer_csr_max_value(2, None), 0);
    // Some(name) = that layer only, matched as a whole path component.
    assert_eq!(cat.layer_csr_max_value(0, Some("counts")), 50);
    assert_eq!(cat.layer_csr_max_value(0, Some("counts_sq")), 80);
    assert_eq!(cat.layer_csr_max_value(0, Some("missing")), 0);
    assert_eq!(cat.layer_csr_max_value(1, Some("counts")), 200);
}

/// The single-modality legacy naming (`{layer}_shard_{idx}`, what
/// `layer_names()` parses and every non-multimodal writer emits) must resolve
/// too. It did not: the named fold delegated to
/// `layer_csr_shards_for_modality`, whose `/{layer}/` needle no legacy name
/// contains, so `layer_csr_max_value(0, Some(name))` answered 0 for every
/// single-modality file — indistinguishable from "no large values here", which
/// is what a decode-loss guard reads it as.
#[test]
fn layer_csr_max_value_resolves_the_legacy_single_modality_naming() {
    let cat = catalog_with(vec![
        max_value_entry("counts_shard_0", SectionType::LayerCsrShard, 0, Some(50)),
        max_value_entry("counts_shard_1", SectionType::LayerCsrShard, 0, Some(70)),
        // Trap: shares the `counts` prefix but is a different layer, and
        // `starts_with("counts_shard_")` must not match it.
        max_value_entry("counts_sq_shard_0", SectionType::LayerCsrShard, 0, Some(90)),
        max_value_entry("X_shard_0", SectionType::CsrShard, 0, Some(999)),
    ]);
    // The delegate the fold used to call still cannot see these entries — this
    // is the whole defect, kept visible so a later "simplify back to one
    // predicate" cannot quietly restore it.
    assert!(cat.layer_csr_shards_for_modality(0, "counts").is_empty());
    assert_eq!(cat.layer_csr_max_value(0, Some("counts")), 70);
    assert_eq!(cat.layer_csr_max_value(0, Some("counts_sq")), 90);
    assert_eq!(cat.layer_csr_max_value(0, Some("missing")), 0);
    // The unnamed fold already covered this naming and still does.
    assert_eq!(cat.layer_csr_max_value(0, None), 90);
    // And the names the fold accepts are exactly the ones `layer_names()`
    // reports, so a caller cannot hand it a name it will silently miss.
    assert_eq!(cat.layer_names(), vec!["counts", "counts_sq"]);
}

/// Both namings resolve through one call, and a mixed catalog does not
/// double-count or cross-contaminate: no writer emits both spellings for one
/// layer, but the fold must be right if it ever sees them side by side.
#[test]
fn layer_csr_shards_named_accepts_either_naming() {
    let cat = catalog_with(vec![
        max_value_entry("counts_shard_0", SectionType::LayerCsrShard, 0, Some(50)),
        max_value_entry(
            "layer/rna/counts/shard_0",
            SectionType::LayerCsrShard,
            0,
            Some(60),
        ),
        // Same layer name under a different modality: never in scope.
        max_value_entry(
            "layer/atac/counts/shard_0",
            SectionType::LayerCsrShard,
            1,
            Some(400),
        ),
    ]);
    assert_eq!(cat.layer_csr_shards_named(0, "counts").len(), 2);
    assert_eq!(cat.layer_csr_max_value(0, Some("counts")), 60);
    assert_eq!(cat.layer_csr_shards_named(1, "counts").len(), 1);
    assert_eq!(cat.layer_csr_max_value(1, Some("counts")), 400);
    assert!(cat.layer_csr_shards_named(0, "nope").is_empty());
}

/// The subset fold a filtered layer read needs: max over the named layers and
/// nothing else, in one pass, under either naming. `layer_csr_max_value(_,
/// Some(name))` is the one-name case of exactly this, so the two cannot
/// disagree.
#[test]
fn layer_csr_max_value_over_folds_only_the_named_layers() {
    let cat = catalog_with(vec![
        max_value_entry("narrow_shard_0", SectionType::LayerCsrShard, 0, Some(50)),
        max_value_entry(
            "wide_shard_0",
            SectionType::LayerCsrShard,
            0,
            Some(20_000_000),
        ),
        max_value_entry(
            "layer/rna/mid/shard_0",
            SectionType::LayerCsrShard,
            0,
            Some(700),
        ),
        max_value_entry(
            "wide_shard_0",
            SectionType::LayerCsrShard,
            1,
            Some(u32::MAX),
        ),
        max_value_entry("X_shard_0", SectionType::CsrShard, 0, Some(999)),
    ]);
    // The whole point: the wide layer does not contribute unless it is named.
    assert_eq!(cat.layer_csr_max_value_over(0, &["narrow"]), 50);
    assert_eq!(cat.layer_csr_max_value_over(0, &["wide"]), 20_000_000);
    assert_eq!(
        cat.layer_csr_max_value_over(0, &["narrow", "wide"]),
        20_000_000
    );
    // Both namings resolve through the one call.
    assert_eq!(cat.layer_csr_max_value_over(0, &["narrow", "mid"]), 700);
    // Nothing selected decodes nothing, so nothing can round.
    assert_eq!(cat.layer_csr_max_value_over(0, &[]), 0);
    // A name the modality does not carry contributes nothing, and another
    // modality's same-named layer is never in scope.
    assert_eq!(cat.layer_csr_max_value_over(0, &["missing"]), 0);
    assert_eq!(cat.layer_csr_max_value_over(2, &["wide"]), 0);
    assert_eq!(cat.layer_csr_max_value_over(1, &["wide"]), u32::MAX);
    // X shards never contribute, whatever is named.
    assert_eq!(cat.layer_csr_max_value_over(0, &["X"]), 0);
    // Equal to the single-name accessor by construction.
    for name in ["narrow", "wide", "mid", "missing"] {
        assert_eq!(
            cat.layer_csr_max_value(0, Some(name)),
            cat.layer_csr_max_value_over(0, &[name]),
            "single-name accessor disagreed for {name}"
        );
    }
    // And naming every layer equals the unfiltered whole-modality fold.
    assert_eq!(
        cat.layer_csr_max_value_over(0, &["narrow", "wide", "mid"]),
        cat.layer_csr_max_value(0, None)
    );
}

/// An entry without `ShardStats` contributes 0 to the fold (it is skipped) —
/// the guard is only as strong as the catalog it reads. Deliberate pre-1.0
/// semantics; every current writer path emits stats.
#[test]
fn value_max_folds_treat_missing_stats_as_zero() {
    // A stats-less entry beside a stats-bearing sibling, in EACH family the
    // folds cover — pinning "skipped", not "no matching section existed".
    let cat = catalog_with(vec![
        max_value_entry("X_shard_0", SectionType::CsrShard, 0, None),
        max_value_entry("X_shard_1", SectionType::CsrShard, 0, Some(7)),
        max_value_entry("raw/X_shard_0", SectionType::RawCsrShard, 0, None),
        max_value_entry("raw/X_shard_1", SectionType::RawCsrShard, 0, Some(11)),
        max_value_entry(
            "layer/rna/counts/shard_0",
            SectionType::LayerCsrShard,
            0,
            None,
        ),
        max_value_entry(
            "layer/rna/counts/shard_1",
            SectionType::LayerCsrShard,
            0,
            Some(13),
        ),
    ]);
    assert_eq!(cat.csr_max_value(None), 7);
    assert_eq!(cat.raw_csr_max_value(), 11);
    assert_eq!(cat.layer_csr_max_value(0, None), 13);
    assert_eq!(cat.layer_csr_max_value(0, Some("counts")), 13);

    // All matching entries stats-less (per family) → 0, not an error.
    let all_statless = catalog_with(vec![
        max_value_entry("X_shard_0", SectionType::CsrShard, 0, None),
        max_value_entry("raw/X_shard_0", SectionType::RawCsrShard, 0, None),
        max_value_entry(
            "layer/rna/counts/shard_0",
            SectionType::LayerCsrShard,
            0,
            None,
        ),
    ]);
    assert_eq!(all_statless.csr_max_value(None), 0);
    assert_eq!(all_statless.raw_csr_max_value(), 0);
    assert_eq!(all_statless.layer_csr_max_value(0, None), 0);
    assert_eq!(all_statless.layer_csr_max_value(0, Some("counts")), 0);

    let empty = catalog_with(vec![]);
    assert_eq!(empty.csr_max_value(None), 0);
    assert_eq!(empty.raw_csr_max_value(), 0);
}
