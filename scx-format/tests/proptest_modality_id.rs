//! Property-based test for catalog `modality_id` round-trip
//! (Phase K.5 of MULTIMODAL-SUPPORT).
//!
//! Generates random `FullCatalogEntry` vectors with arbitrary
//! `modality_id` values, serializes the catalog, and verifies that
//! parsing recovers exactly the same entries — bitwise — through the
//! v2 layout (`catalog_version = CURRENT_CATALOG_VERSION`). Locks the
//! invariant that `modality_id` survives serialize → deserialize for
//! every section type, every name length the layout supports, and
//! the full 0..=255 routing-key range.
//!
//! The proptest covers the same parser surface as a libfuzzer target
//! for the catalog modality byte; the libfuzzer variant
//! (`fuzz_modality_table.rs` per Phase K.5) is deferred to a follow-on
//! since it requires a nightly toolchain.

use proptest::prelude::*;
use scx_format::catalog::{FullCatalog, FullCatalogEntry, CURRENT_CATALOG_VERSION};
use scx_format::SectionType;

/// All valid `SectionType` discriminants. `SectionType::from_u8`
/// rejects anything outside this list, which would silently drop the
/// entry on read-back — so the strategy must only emit values present
/// here. Updating this list when new section types are added keeps
/// the proptest in sync without losing coverage.
const VALID_SECTION_TYPES: &[u8] = &[
    SectionType::ObsMetadata as u8,
    SectionType::ObsIndex as u8,
    SectionType::VarMetadata as u8,
    SectionType::VarIndex as u8,
    SectionType::CsrShard as u8,
    SectionType::CscShard as u8,
    SectionType::BitmapShard as u8,
    SectionType::LayerCsrShard as u8,
    SectionType::ObsmEmbedding as u8,
    SectionType::ObspCsrShard as u8,
    SectionType::UnsBlob as u8,
    SectionType::Provenance as u8,
    SectionType::DeletionVectors as u8,
    SectionType::ObsPredicateIndex as u8,
    SectionType::VarPredicateIndex as u8,
    SectionType::ModalityTable as u8,
    SectionType::LayerCscShard as u8,
];

prop_compose! {
    fn arb_entry()(
        modality_id in 0u8..=255,
        section_type_idx in 0..VALID_SECTION_TYPES.len(),
        offset in 0u64..u64::MAX / 2,
        length in 0u64..u64::MAX / 2,
        name in "[A-Za-z0-9_]{0,64}",
        checksum_seed in any::<[u8; 32]>(),
    ) -> FullCatalogEntry {
        let raw = VALID_SECTION_TYPES[section_type_idx];
        let section_type = SectionType::from_u8(raw)
            .expect("VALID_SECTION_TYPES is the from_u8 inverse");
        FullCatalogEntry {
            name,
            offset,
            length,
            section_type,
            checksum: checksum_seed,
            modality_id,
            stats: None,
        }
    }
}

fn arb_catalog() -> impl Strategy<Value = FullCatalog> {
    (
        0u64..=1024,
        0u64..u64::MAX / 2,
        0u64..u64::MAX / 2,
        prop::collection::vec(arb_entry(), 0..=32),
    )
        .prop_map(
            |(n_obs, manifest_sequence, prev_catalog_offset, entries)| FullCatalog {
                catalog_version: CURRENT_CATALOG_VERSION,
                manifest_sequence,
                prev_catalog_offset,
                n_obs,
                entries,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Every entry's `modality_id` survives a v2 catalog write→read.
    /// Also verifies the rest of the entry payload (name, offset,
    /// length, section_type, checksum) — a round-trip regression
    /// guard for the per-entry layout.
    #[test]
    fn full_catalog_round_trip_preserves_modality_id(cat in arb_catalog()) {
        let mut buf = Vec::new();
        cat.write_to(&mut buf).expect("write_to should succeed");

        let mut cur = std::io::Cursor::new(&buf[..]);
        let parsed = FullCatalog::read_from(&mut cur, buf.len(), true)
            .expect("read_from should succeed");

        prop_assert_eq!(parsed.catalog_version, cat.catalog_version);
        prop_assert_eq!(parsed.manifest_sequence, cat.manifest_sequence);
        prop_assert_eq!(parsed.prev_catalog_offset, cat.prev_catalog_offset);
        prop_assert_eq!(parsed.n_obs, cat.n_obs);
        prop_assert_eq!(parsed.entries.len(), cat.entries.len());

        for (a, b) in cat.entries.iter().zip(&parsed.entries) {
            prop_assert_eq!(a.modality_id, b.modality_id);
            prop_assert_eq!(&a.name, &b.name);
            prop_assert_eq!(a.offset, b.offset);
            prop_assert_eq!(a.length, b.length);
            prop_assert_eq!(a.section_type as u8, b.section_type as u8);
            prop_assert_eq!(a.checksum, b.checksum);
            prop_assert!(b.stats.is_none());
        }
    }

    /// Targeted coverage of `modality_id == 0` (global / single-modality
    /// default) and the full 1..=255 range. Exercises the boundary
    /// conditions that real multimodal files will hit (1..=n_modalities
    /// for registered modalities, plus 0 for the global obs/uns
    /// entries).
    #[test]
    fn modality_id_full_range_round_trip(modality_id in 0u8..=255) {
        let cat = FullCatalog {
            catalog_version: CURRENT_CATALOG_VERSION,
            manifest_sequence: 0,
            prev_catalog_offset: 0,
            n_obs: 100,
            entries: vec![FullCatalogEntry {
                name: "X_shard_0".to_string(),
                offset: 4352,
                length: 1024,
                section_type: SectionType::CsrShard,
                checksum: [0xAB; 32],
                modality_id,
                stats: None,
            }],
        };

        let mut buf = Vec::new();
        cat.write_to(&mut buf).unwrap();
        let mut cur = std::io::Cursor::new(&buf[..]);
        let parsed = FullCatalog::read_from(&mut cur, buf.len(), true).unwrap();

        prop_assert_eq!(parsed.entries.len(), 1);
        prop_assert_eq!(parsed.entries[0].modality_id, modality_id);
    }
}
