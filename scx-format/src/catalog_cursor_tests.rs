use super::*;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::catalog::{FullCatalog, FullCatalogEntry, ShardStats, CURRENT_CATALOG_VERSION};
use crate::section::SectionType;

fn entry(name: &str, section_type: SectionType, stats: Option<ShardStats>) -> FullCatalogEntry {
    FullCatalogEntry {
        name: name.to_string(),
        offset: 4352,
        length: 1000,
        section_type,
        checksum: [0x5A; 32],
        modality_id: 3,
        stats,
    }
}

fn csr_stats(row_start: u64, row_end: u64) -> ShardStats {
    ShardStats {
        row_start,
        row_end,
        col_start: 0,
        col_end: 500,
        nnz: 1234,
        value_min: 1,
        value_max: 9,
        value_sum: 4321,
        n_indexed_columns: 0,
        column_stats: Vec::new(),
    }
}

fn serialized(entries: Vec<FullCatalogEntry>, generations: (u64, u64)) -> Vec<u8> {
    serialized_as(CURRENT_CATALOG_VERSION, entries, generations)
}

fn serialized_as(
    catalog_version: u16,
    entries: Vec<FullCatalogEntry>,
    generations: (u64, u64),
) -> Vec<u8> {
    let catalog = FullCatalog {
        catalog_version,
        manifest_sequence: 7,
        prev_catalog_offset: 99,
        n_obs: 640,
        entries,
        data_generation: generations.0,
        csc_build_generation: generations.1,
    };
    let mut buf = Vec::new();
    catalog.write_to(&mut buf).unwrap();
    buf
}

/// Every field the writer emitted comes back off the cursor unchanged, and
/// the borrowed payloads are the exact sub-slices — not copies sized by a
/// second reading of the length prefixes.
#[test]
fn yields_every_field_the_writer_emitted() {
    let bytes = serialized(
        vec![
            entry("obs", SectionType::ObsMetadata, None),
            entry("X_shard_0", SectionType::CsrShard, Some(csr_stats(0, 64))),
        ],
        (11, 11),
    );

    let mut cursor = CatalogEntryCursor::new(&bytes, true).unwrap();
    assert_eq!(cursor.preamble().catalog_version, CURRENT_CATALOG_VERSION);
    assert_eq!(cursor.preamble().manifest_sequence, 7);
    assert_eq!(cursor.preamble().prev_catalog_offset, 99);
    assert_eq!(cursor.preamble().n_obs, 640);
    assert_eq!(cursor.preamble().n_entries, 2);

    let first = cursor.next_entry().unwrap().unwrap();
    assert_eq!(first.name_bytes, b"obs");
    assert_eq!(first.offset, 4352);
    assert_eq!(first.length, 1000);
    assert_eq!(first.section_type_raw, SectionType::ObsMetadata as u8);
    assert_eq!(first.checksum_bytes, &[0x5A; 32]);
    assert_eq!(first.modality_id, 3);
    assert!(first.stats_bytes.is_empty(), "no stats → empty, not absent");

    let second = cursor.next_entry().unwrap().unwrap();
    assert_eq!(second.name_bytes, b"X_shard_0");
    assert_eq!(second.section_type_raw, SectionType::CsrShard as u8);
    let mut reader: &[u8] = second.stats_bytes;
    let decoded = ShardStats::read_from(&mut reader, CURRENT_CATALOG_VERSION).unwrap();
    assert_eq!(decoded, csr_stats(0, 64));

    assert!(cursor.next_entry().is_none(), "declared count is the bound");
    assert_eq!(cursor.finish().unwrap(), (11, 11));
}

/// The v4 trailer is the cursor's, not each consumer's. `CatalogView` used to
/// have no way to report it at all (§4.7).
#[test]
fn trailer_is_zero_below_v4_and_read_above() {
    // `write_to` floors the declared version at 2 and emits the trailer only
    // at 4, so declaring 2 with zero counters yields a catalog with no
    // trailing bytes at all.
    let bytes = serialized_as(
        2,
        vec![entry("obs", SectionType::ObsMetadata, None)],
        (0, 0),
    );
    let cursor = CatalogEntryCursor::new(&bytes, true).unwrap();
    assert_eq!(cursor.preamble().catalog_version, 2);
    let mut cursor = cursor;
    while cursor.next_entry().is_some() {}
    assert_eq!(cursor.finish().unwrap(), (0, 0));

    let v4 = serialized(vec![entry("obs", SectionType::ObsMetadata, None)], (42, 41));
    let mut cursor = CatalogEntryCursor::new(&v4, true).unwrap();
    assert_eq!(cursor.preamble().catalog_version, 4);
    while cursor.next_entry().is_some() {}
    assert_eq!(cursor.finish().unwrap(), (42, 41));
}

/// Taking the trailer before the entry list is drained would read entry bytes
/// as counters. That is a caller bug, so it is an error rather than a silently
/// wrong `(data_generation, csc_build_generation)`.
#[test]
fn finish_before_draining_is_an_error() {
    let bytes = serialized(
        vec![
            entry("obs", SectionType::ObsMetadata, None),
            entry("var", SectionType::VarMetadata, None),
        ],
        (5, 5),
    );
    let mut cursor = CatalogEntryCursor::new(&bytes, true).unwrap();
    cursor.next_entry().unwrap().unwrap();
    let err = cursor.finish().unwrap_err();
    assert!(
        format!("{err}").contains("1 entries still unread"),
        "expected an undrained-cursor error, got: {err}"
    );
}

/// Checksum semantics are the cursor's now, so both parsers inherit them
/// identically: verify rejects a flipped payload byte, and `verify=false`
/// accepts it.
#[test]
fn checksum_is_verified_at_construction() {
    let bytes = serialized(vec![entry("obs", SectionType::ObsMetadata, None)], (0, 0));
    CatalogEntryCursor::new(&bytes, true).unwrap();

    let mut corrupt = bytes.clone();
    corrupt[10] ^= 0xFF;
    let err = CatalogEntryCursor::new(&corrupt, true).unwrap_err();
    assert!(matches!(err, ScxError::ChecksumMismatch { .. }));
    CatalogEntryCursor::new(&corrupt, false).unwrap();

    let err = CatalogEntryCursor::new(&[0u8; 8], true).unwrap_err();
    assert!(matches!(err, ScxError::ChecksumMismatch { .. }));
}

/// A declared count that could not fit even at the minimum entry size is
/// rejected before it drives a `Vec::with_capacity` or a long walk.
#[test]
fn absurd_entry_count_is_rejected_up_front() {
    let mut payload = Vec::new();
    payload.write_u16::<LittleEndian>(2).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u64::<LittleEndian>(0).unwrap();
    payload.write_u32::<LittleEndian>(u32::MAX).unwrap();
    payload.extend_from_slice(&[0u8; 32]);

    let err = CatalogEntryCursor::new(&payload, false).unwrap_err();
    assert!(
        format!("{err}").contains("allocation too large"),
        "expected an allocation guard, got: {err}"
    );
}

/// `catalog_payload_len` is what a caller uses when it does not yet know
/// where the catalog ends — `scx info` walking the manifest chain. It must
/// land exactly on the checksum, and it must do so with the slice running
/// past the catalog, which is the shape a mmap hands it.
#[test]
fn payload_len_finds_the_checksum_boundary() {
    for generations in [(0, 0), (3, 3)] {
        let bytes = serialized(
            vec![
                entry("obs", SectionType::ObsMetadata, None),
                entry("X_shard_0", SectionType::CsrShard, Some(csr_stats(0, 64))),
                entry("X_shard_1", SectionType::CsrShard, Some(csr_stats(64, 128))),
            ],
            generations,
        );

        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&[0xEE; 512]); // bytes past the catalog

        let payload_len = catalog_payload_len(&trailing).unwrap();
        assert_eq!(
            payload_len + CATALOG_CHECKSUM_LEN,
            bytes.len(),
            "computed length must land on the catalog's own end for {generations:?}"
        );

        // And the length it computes is one `FullCatalog::read_from` accepts.
        FullCatalog::read_from(
            &mut std::io::Cursor::new(&trailing),
            payload_len + CATALOG_CHECKSUM_LEN,
            true,
        )
        .unwrap();
    }
}

/// The per-entry arithmetic lives in one place; these constants are what the
/// two `File`-based sizing walks (`scx-ops`'s rollback reader) consume
/// instead of keeping their own copies.
#[test]
fn layout_constants_match_the_writer() {
    assert_eq!(CATALOG_PREAMBLE_LEN, 30);
    assert_eq!(CATALOG_CHECKSUM_LEN, 32);
    // v1: name_len(2) + offset(8) + length(8) + type(1) + checksum(32)
    //     + stats_len(2), with an empty name.
    assert_eq!(MIN_ENTRY_BYTES, 2 + 49 + 2);
    assert_eq!(entry_fixed_bytes_after_name(1), 49);
    assert_eq!(entry_fixed_bytes_after_name(2), 50);
    assert_eq!(entry_fixed_bytes_after_name(4), 50);
    assert_eq!(v4_trailer_len(3), 0);
    assert_eq!(v4_trailer_len(4), 16);

    // Cross-check against a real serialized catalog rather than restating
    // the arithmetic: one entry with an empty name and no stats.
    for version in [2u16, 4] {
        let bytes = serialized_as(
            version,
            vec![entry("", SectionType::ObsMetadata, None)],
            (0, 0),
        );
        let per_entry = 2 + entry_fixed_bytes_after_name(version) + 2;
        assert_eq!(
            bytes.len(),
            CATALOG_PREAMBLE_LEN + per_entry + v4_trailer_len(version) + CATALOG_CHECKSUM_LEN,
            "layout arithmetic disagrees with the writer at v{version}"
        );
    }
}
