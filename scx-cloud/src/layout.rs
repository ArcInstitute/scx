//! Shared packed-file layout helpers for the cloud write paths
//! (`pull`, `pack`, `cloud_optimize`).
//!
//! These three operations all rewrite an SCX file with a cloud-optimized
//! section ordering and a front-of-file catalog. The section ordering,
//! root-catalog construction, catalog-size estimate, and whole-file checksum
//! are identical across all three; they live here so a change lands on every
//! path at once.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

use scx_format_io::catalog::{FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry};
use scx_format_io::section::SectionType;

use crate::error::Result;

/// Offset where sections begin: 256 (header) + 4096 (root catalog placeholder).
///
/// Re-exported from `scx_format_io` so the cloud write paths share the single
/// canonical definition.
pub(crate) use scx_format_io::SECTIONS_START_OFFSET;

/// Cloud-optimized section ordering for the rewritten file.
///
/// `CscShard` is placed adjacent to `CsrShard` so column-major reads stay in
/// the contiguous prefix region of the packed file. Section types not listed
/// here (unknown/future types) are appended after, in `section_type` order.
pub(crate) const SECTION_ORDER: &[SectionType] = &[
    SectionType::ObsMetadata,
    SectionType::ObsIndex,
    SectionType::VarMetadata,
    SectionType::VarIndex,
    SectionType::CsrShard,
    SectionType::CscShard,
    SectionType::LayerCsrShard,
    SectionType::ObsmEmbedding,
    SectionType::ObspCsrShard,
    SectionType::UnsBlob,
    SectionType::ObsPredicateIndex,
    SectionType::VarPredicateIndex,
    SectionType::Provenance,
    SectionType::DeletionVectors,
];

/// Order catalog entries into the cloud-optimized layout: group by section
/// type, emit groups in [`SECTION_ORDER`], then append any section types not
/// in that list (unknown/future types) in `section_type` order. Order within
/// a group is preserved.
pub(crate) fn order_entries_for_layout(entries: &[FullCatalogEntry]) -> Vec<&FullCatalogEntry> {
    let mut grouped: BTreeMap<u8, Vec<&FullCatalogEntry>> = BTreeMap::new();
    for entry in entries {
        grouped
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    let mut ordered: Vec<&FullCatalogEntry> = Vec::with_capacity(entries.len());
    for &st in SECTION_ORDER {
        if let Some(group) = grouped.remove(&(st as u8)) {
            ordered.extend(group);
        }
    }
    // Whatever remains is an unknown/future section type — appended in
    // `section_type` order (`BTreeMap` iterates by sorted key).
    for group in grouped.into_values() {
        ordered.extend(group);
    }
    ordered
}

/// Estimate the serialized size of a full catalog with `n_entries` entries,
/// rounded up to 8-byte alignment (30-byte header + 100 bytes/entry + 32-byte
/// trailing checksum).
pub(crate) fn estimate_catalog_size(n_entries: usize) -> usize {
    let size = 30 + n_entries * 100 + 32;
    (size + 7) & !7
}

/// Build a root catalog (per-section-type group summary) from a full catalog.
pub(crate) fn build_root_catalog(catalog: &FullCatalog) -> RootCatalog {
    let mut groups: BTreeMap<u8, Vec<&FullCatalogEntry>> = BTreeMap::new();
    for entry in &catalog.entries {
        groups
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    let mut root_entries = Vec::new();
    for (&group_type, entries) in &groups {
        let first_offset = entries.iter().map(|e| e.offset).min().unwrap_or(0);
        let total_length: u64 = entries.iter().map(|e| e.length).sum();
        let n_sections = entries.len() as u32;
        root_entries.push(RootCatalogEntry {
            group_type,
            first_section_offset: first_offset,
            total_group_length: total_length,
            n_sections,
            summary: [0u8; 32],
        });
    }

    RootCatalog {
        n_section_groups: root_entries.len() as u16,
        entries: root_entries,
    }
}

/// Compute file checksum: BLAKE3 of the entire file, truncated to u64.
pub(crate) fn compute_file_checksum(file: &mut (impl Read + Seek)) -> Result<u64> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    let hash = hasher.finalize();
    Ok(scx_format_io::checksum::truncate_hash_to_u64(&hash))
}
