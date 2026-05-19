//! Property-based test for `ModalityTable` write→read round-trip
//!
//! Complements `proptest_modality_id.rs`, which exercises the catalog
//! routing-key byte. This suite covers the full `ModalityTable` section
//! payload: random `Vec<ModalityInfo>` with varied types, name lengths
//! (1..=64 bytes), flag bitmaps, and per-modality counts.
//!
//! Invariant: write_to → read_from is the identity, bitwise, for any
//! collection of valid `ModalityInfo` entries with unique names.

use proptest::prelude::*;
use scx_format::{ModalityFlags, ModalityInfo, ModalityTable, ModalityType};

/// Strategy for a single `ModalityType`.
fn arb_modality_type() -> impl Strategy<Value = ModalityType> {
    prop_oneof![
        Just(ModalityType::Rna),
        Just(ModalityType::Protein),
        Just(ModalityType::Atac),
        Just(ModalityType::Spatial),
        Just(ModalityType::Methylation),
        Just(ModalityType::Custom),
    ]
}

/// Strategy for a `ModalityFlags` value covering every defined bit.
fn arb_modality_flags() -> impl Strategy<Value = ModalityFlags> {
    (0u8..=0b0011_1111).prop_map(ModalityFlags::from_bits_truncate)
}

/// Strategy for a single `ModalityInfo`. The `name` is parameterised
/// by an external `index` (caller supplies the disambiguating suffix)
/// so that the table-level strategy can enforce uniqueness without
/// rejection sampling.
fn arb_modality_info(index: usize) -> impl Strategy<Value = ModalityInfo> {
    (
        // Base name (1..32 bytes of ASCII letters/digits). Combined with
        // the index suffix below, the total fits inside the 64-byte cap.
        "[a-zA-Z][a-zA-Z0-9_]{0,28}",
        arb_modality_type(),
        any::<u8>(), // default_codec_id
        any::<u8>(), // default_value_encoding
        0u64..=10_000_000,
        0u64..=10_000_000_000,
        0u32..=4096,
        0u32..=4096,
        arb_modality_flags(),
    )
        .prop_map(
            move |(
                base,
                modality_type,
                default_codec_id,
                default_value_encoding,
                n_vars,
                nnz,
                n_csr_shards,
                n_csc_shards,
                flags,
            )| ModalityInfo {
                name: format!("{base}_{index}"),
                modality_type,
                default_codec_id,
                default_value_encoding,
                n_vars,
                nnz,
                n_csr_shards,
                n_csc_shards,
                flags,
            },
        )
}

/// Strategy for a `ModalityTable` with 0..=8 entries, each name made
/// unique via its positional suffix.
fn arb_modality_table() -> impl Strategy<Value = ModalityTable> {
    (0usize..=8).prop_flat_map(|n| {
        let entry_strats: Vec<_> = (0..n).map(|i| arb_modality_info(i).boxed()).collect();
        entry_strats.prop_map(ModalityTable::new)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// `ModalityTable::write_to` → `read_from` is the identity for any
    /// valid table with unique names.
    #[test]
    fn modality_table_round_trip(table in arb_modality_table()) {
        let mut buf = Vec::new();
        table.write_to(&mut buf).expect("write_to should succeed");

        prop_assert_eq!(buf.len(), table.serialized_len(),
            "serialized_len must match actual write length");

        let mut cur = std::io::Cursor::new(&buf);
        let decoded = ModalityTable::read_from(&mut cur, buf.len())
            .expect("read_from should succeed");

        prop_assert_eq!(decoded.entries.len(), table.entries.len());
        for (orig, parsed) in table.entries.iter().zip(decoded.entries.iter()) {
            prop_assert_eq!(&orig.name, &parsed.name);
            prop_assert_eq!(orig.modality_type as u8, parsed.modality_type as u8);
            prop_assert_eq!(orig.default_codec_id, parsed.default_codec_id);
            prop_assert_eq!(orig.default_value_encoding, parsed.default_value_encoding);
            prop_assert_eq!(orig.n_vars, parsed.n_vars);
            prop_assert_eq!(orig.nnz, parsed.nnz);
            prop_assert_eq!(orig.n_csr_shards, parsed.n_csr_shards);
            prop_assert_eq!(orig.n_csc_shards, parsed.n_csc_shards);
            prop_assert_eq!(orig.flags.bits(), parsed.flags.bits());
        }
    }

    /// Single-modality table covering every `ModalityType` variant.
    #[test]
    fn modality_type_round_trip(
        modality_type in arb_modality_type(),
        flags in arb_modality_flags(),
    ) {
        let entry = ModalityInfo {
            name: "test".to_string(),
            modality_type,
            default_codec_id: 1,
            default_value_encoding: 0,
            n_vars: 100,
            nnz: 1000,
            n_csr_shards: 1,
            n_csc_shards: 0,
            flags,
        };
        let table = ModalityTable::new(vec![entry.clone()]);

        let mut buf = Vec::new();
        table.write_to(&mut buf).unwrap();
        let mut cur = std::io::Cursor::new(&buf);
        let decoded = ModalityTable::read_from(&mut cur, buf.len()).unwrap();

        prop_assert_eq!(decoded.entries.len(), 1);
        prop_assert_eq!(decoded.entries[0].modality_type as u8, modality_type as u8);
        prop_assert_eq!(decoded.entries[0].flags.bits(), flags.bits());
    }
}
