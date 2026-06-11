//! Lightweight reader-facing catalog representation.
//!
//! Drops the per-entry fields that the read/open path doesn't touch
//! (the 32-byte BLAKE3 `checksum`, the diagnostic `value_min` /
//! `value_max` / `value_sum`, the unused-on-the-hot-path
//! `col_start` / `col_end` on row-major shards, and the predicate-only
//! `column_stats`). For row-major shard reader construction
//! (`BackedCsrReader`, `BackedCsrIndex`) the only stats fields needed
//! are the major-axis range and `nnz`; for layer prefix filtering the
//! name is needed, but for ordinary X-shard construction it is not.
//!
//! `CatalogView` integrates with `BackedCsrReader::new*` (in the
//! `scx-format-io` crate); see [`crate::catalog::FullCatalog`] for the
//! full eagerly-parsed representation used by validation, mutation,
//! and `scx-engine` predicate pushdown. The `from_full` constructor
//! is exposed primarily for tests and for callers that hold a
//! `FullCatalog` from a non-byte source.
//!
//! The parser is intentionally a direct read of the catalog payload
//! slice — no per-entry `vec![0u8; name_len]`, no per-entry
//! `String::from_utf8` for shard entries, and no second `Cursor`
//! wrapper around stats bytes (matching the same fixes that landed
//! in `FullCatalog::read_from`).

use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::catalog::FullCatalog;
use crate::checksum::blake3_hash;
use crate::error::{Result, ScxError};
use crate::section::SectionType;

/// Major-axis-only shard statistics for the read/open path. Row range
/// for row-major shards (CSR / Layer-CSR / Obsp-CSR); column range for
/// column-major shards (CSC / Layer-CSC). Stored in dispatched form so
/// callers don't need to re-branch on `section_type` per access.
///
/// 24 bytes / entry vs. the ≥73-byte `ShardStats` it replaces on the
/// hot path. The diagnostic / pushdown-only fields the full struct
/// carries (`value_min` / `value_max` / `value_sum`,
/// `n_indexed_columns`, `column_stats`) are dropped here; callers that
/// need them go through [`crate::catalog::LazyShardStats`] (lazy
/// `column_stats` decode) or [`crate::catalog::ShardStats`] (eager).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardStatsLite {
    /// Major-axis start. For row-major shards: `row_start`. For v2
    /// column-major shards: `col_start`. For v1 column-major shards
    /// (axis-overloaded): the legacy `row_start` slot, which carries
    /// the column index.
    pub major_start: u64,
    /// Major-axis end. Symmetric with `major_start`.
    pub major_end: u64,
    /// Non-zero count for budget auto-tune in `IndexPlanLoader`.
    pub nnz: u64,
}

/// Lightweight, immutable catalog entry. ~24 bytes of stats + ~24
/// bytes of fixed metadata + an optional name `Box<str>` — vs.
/// `FullCatalogEntry`'s 32-byte `checksum` + full `ShardStats` + always
/// allocated `String` name.
///
/// `name` is `Some` only for entries whose section type may be looked
/// up by name (metadata, layer prefix filtering). For pure CSR / CSC /
/// Obsp CSR shard entries the name carries no read-path information,
/// so it is dropped at parse time. Callers that need full names should
/// go through `FullCatalog` instead.
#[derive(Debug, Clone)]
pub struct CatalogViewEntry {
    /// Section name. `None` for `CsrShard` / `CscShard` /
    /// `ObspCsrShard` entries — the read path indexes those by stats,
    /// not name. `Some(_)` for layer shards and non-shard entries.
    pub name: Option<Box<str>>,
    /// File offset of this section.
    pub offset: u64,
    /// Byte length of this section.
    pub length: u64,
    pub section_type: SectionType,
    /// Modality routing key. `0` for v1 / single-modality v2 files.
    pub modality_id: u8,
    /// `None` for non-shard entries and for shard entries with
    /// `stats_len == 0` on disk. `Some(_)` carries the dispatched
    /// major-axis range and `nnz`.
    pub stats: Option<ShardStatsLite>,
}

impl CatalogViewEntry {
    /// Borrow the entry's name. `None` for shard entries where the
    /// view dropped the name at parse time.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Lightweight catalog. Mirrors `FullCatalog`'s preamble fields
/// (`catalog_version`, `manifest_sequence`, `prev_catalog_offset`,
/// `n_obs`) and replaces the heavyweight `entries` list with
/// `CatalogViewEntry`. Designed to be wrapped in `Arc<CatalogView>` so
/// multiple `ScxReader` instances within one process can share a
/// single parsed catalog without mutating it.
#[derive(Debug, Clone)]
pub struct CatalogView {
    pub catalog_version: u16,
    pub manifest_sequence: u64,
    pub prev_catalog_offset: u64,
    pub n_obs: u64,
    pub entries: Vec<CatalogViewEntry>,
}

/// Whether the lightweight view should retain a string copy of the
/// section name. Pure row/column shard entries do not — their
/// read-path consumers (`BackedCsrIndex`, `BackedCscReader`) index
/// strictly by stats. Anything that may be looked up by name (layer
/// prefix filter, `catalog.get("obs")`) does.
fn name_required_for(section_type: SectionType) -> bool {
    !matches!(
        section_type,
        SectionType::CsrShard | SectionType::CscShard | SectionType::ObspCsrShard
    )
}

/// Pick the dispatched major-axis byte offsets inside a stats payload.
/// Returns `(major_start_off, major_end_off, nnz_off)`. v1 stats are
/// 41 bytes (no `col_start`/`col_end`); v2 stats are 57 bytes
/// (`col_start`/`col_end` injected after `row_end`). On v1 catalogs,
/// `CscShard` entries are axis-overloaded — the legacy `row_*` slot
/// carries the column range, so the offsets are the same as for
/// row-major shards (the v1 reconciliation that `FullCatalog::read_from`
/// performs is implicit here).
fn stats_axis_offsets(catalog_version: u16, section_type: SectionType) -> (usize, usize, usize) {
    let is_column_major = matches!(
        section_type,
        SectionType::CscShard | SectionType::LayerCscShard
    );
    if catalog_version >= 2 && is_column_major {
        // v2 CSC layout: row_start(8) row_end(8) col_start(8) col_end(8) nnz(8) ...
        (16, 24, 32)
    } else if catalog_version >= 2 {
        // v2 row-major: row_start(8) row_end(8) col_start(8) col_end(8) nnz(8) ...
        (0, 8, 32)
    } else {
        // v1 (any section_type): row_start(8) row_end(8) nnz(8) ...
        // For v1 CSC the row pair is axis-overloaded and carries the
        // column range — semantically the major axis either way.
        (0, 8, 16)
    }
}

/// Parse `(major_start, major_end, nnz)` from a stats payload slice
/// dispatched on `(catalog_version, section_type)`. Bytes past the
/// last needed field are ignored — `stats_len` is the authoritative
/// bound, so any extras are forward-compat padding.
fn parse_stats_lite(
    stats_bytes: &[u8],
    catalog_version: u16,
    section_type: SectionType,
) -> Result<ShardStatsLite> {
    let (ms_off, me_off, nnz_off) = stats_axis_offsets(catalog_version, section_type);
    let end = nnz_off + 8;
    if stats_bytes.len() < end {
        return Err(ScxError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "stats payload too short for {:?} v{}: have {}, need >= {}",
                section_type,
                catalog_version,
                stats_bytes.len(),
                end,
            ),
        )));
    }
    let read_u64 = |off: usize| -> u64 {
        u64::from_le_bytes(stats_bytes[off..off + 8].try_into().expect("8 bytes"))
    };
    Ok(ShardStatsLite {
        major_start: read_u64(ms_off),
        major_end: read_u64(me_off),
        nnz: read_u64(nnz_off),
    })
}

impl CatalogView {
    /// Parse a `CatalogView` directly from a catalog payload slice
    /// shaped like `FullCatalog::read_from`'s input — preamble +
    /// entries + 32-byte trailing BLAKE3 checksum. Avoids the per-entry
    /// `String` and stats-payload allocations that `FullCatalog` still
    /// pays for callers that only need read-path metadata.
    ///
    /// When `verify_checksum` is true the trailing 32-byte BLAKE3
    /// hash is validated against the payload prefix. Mirrors
    /// `FullCatalog::read_from` semantics so the lightweight path can
    /// be substituted in `ScxReader::open` without changing reader
    /// trust assumptions.
    pub fn read_from_bytes(payload_with_checksum: &[u8], verify_checksum: bool) -> Result<Self> {
        let total_len = payload_with_checksum.len();
        if total_len < 32 {
            return Err(ScxError::ChecksumMismatch {
                section: "full_catalog (too short)".to_string(),
            });
        }
        let payload_len = total_len - 32;
        let (payload, expected_checksum) = payload_with_checksum.split_at(payload_len);

        if verify_checksum {
            let computed = blake3_hash(payload);
            if computed[..] != *expected_checksum {
                return Err(ScxError::ChecksumMismatch {
                    section: "full_catalog".to_string(),
                });
            }
        }

        let mut cur: &[u8] = payload;
        let catalog_version = cur.read_u16::<LittleEndian>()?;
        let manifest_sequence = cur.read_u64::<LittleEndian>()?;
        let prev_catalog_offset = cur.read_u64::<LittleEndian>()?;
        let n_obs = cur.read_u64::<LittleEndian>()?;
        let n_entries = cur.read_u32::<LittleEndian>()? as usize;

        // Same defensive cap as `FullCatalog::read_from`. The minimum
        // serialised entry size is 53 bytes (v1 lacks modality_id);
        // reject any `n_entries` that couldn't possibly fit.
        const MIN_ENTRY_BYTES: usize = 53;
        crate::error::validate_allocation(n_entries.saturating_mul(MIN_ENTRY_BYTES), payload_len)?;

        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            let name_len = cur.read_u16::<LittleEndian>()? as usize;
            let (name_bytes, rest) = cur.split_at_checked(name_len).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "catalog entry name truncated",
                )
            })?;
            cur = rest;

            let offset = cur.read_u64::<LittleEndian>()?;
            let length = cur.read_u64::<LittleEndian>()?;
            let section_type_raw = cur.read_u8()?;

            // Skip the 32-byte BLAKE3 entry checksum without copying.
            // The lightweight view doesn't retain it — it's the largest
            // per-entry field that the read path never looks at
            // (32 bytes saved per entry).
            let (_skip, rest) = cur.split_at_checked(32).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "catalog entry checksum truncated",
                )
            })?;
            cur = rest;

            let modality_id = if catalog_version >= 2 {
                cur.read_u8()?
            } else {
                0u8
            };

            let stats_len = cur.read_u16::<LittleEndian>()? as usize;
            let (stats_bytes, rest) = cur.split_at_checked(stats_len).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "catalog entry stats payload truncated",
                )
            })?;
            cur = rest;

            // Forward-compat: unknown section types are dropped from
            // the view (matches the warn-and-skip behaviour of
            // `FullCatalog::read_from`).
            let section_type = match SectionType::from_u8(section_type_raw) {
                Some(st) => st,
                None => {
                    log::warn!(
                        "skipping unknown section type {} at offset {}",
                        section_type_raw,
                        offset,
                    );
                    continue;
                }
            };

            let stats = if stats_len > 0 {
                Some(parse_stats_lite(
                    stats_bytes,
                    catalog_version,
                    section_type,
                )?)
            } else {
                None
            };

            // Allocate a name only when the read path could plausibly
            // need it. CSR / CSC / Obsp CSR shards are looked up by
            // (section_type, major_start) — see `BackedCsrIndex` and
            // `BackedCscReader` — so dropping the name on those entries
            // is free for the dominant case (X-shards, the source of
            // the 16K-entry catalog amplification).
            let name = if name_required_for(section_type) {
                Some(
                    std::str::from_utf8(name_bytes)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                        .into(),
                )
            } else {
                None
            };

            entries.push(CatalogViewEntry {
                name,
                offset,
                length,
                section_type,
                modality_id,
                stats,
            });
        }

        Ok(Self {
            catalog_version,
            manifest_sequence,
            prev_catalog_offset,
            n_obs,
            entries,
        })
    }

    /// Build a `CatalogView` from an already-parsed `FullCatalog`. Used
    /// by tests and by callers that hold a `FullCatalog` from a
    /// non-byte source (writer round-trips, fixture builders) and
    /// need to drive the lightweight read paths through it.
    ///
    /// On the hot path prefer `read_from_bytes` — it skips the
    /// `String` and `ShardStats` allocations that `FullCatalog`'s
    /// parser still pays.
    pub fn from_full(full: &FullCatalog) -> Self {
        let entries = full
            .entries
            .iter()
            .map(|e| {
                let stats = e.stats.as_ref().map(|s| {
                    let (major_start, major_end) = match e.section_type {
                        // v2 CSC: explicit col pair. v1 CSC: row pair
                        // already reconciled into col pair by
                        // `FullCatalog::read_from`, so reading col_*
                        // here is correct for both versions.
                        SectionType::CscShard | SectionType::LayerCscShard => {
                            (s.col_start, s.col_end)
                        }
                        _ => (s.row_start, s.row_end),
                    };
                    ShardStatsLite {
                        major_start,
                        major_end,
                        nnz: s.nnz,
                    }
                });
                CatalogViewEntry {
                    name: if name_required_for(e.section_type) {
                        Some(e.name.clone().into_boxed_str())
                    } else {
                        None
                    },
                    offset: e.offset,
                    length: e.length,
                    section_type: e.section_type,
                    modality_id: e.modality_id,
                    stats,
                }
            })
            .collect();
        Self {
            catalog_version: full.catalog_version,
            manifest_sequence: full.manifest_sequence,
            prev_catalog_offset: full.prev_catalog_offset,
            n_obs: full.n_obs,
            entries,
        }
    }

    /// Linear lookup by name. Matches `FullCatalog::get`. Returns
    /// `None` for entries the view dropped the name on (pure shard
    /// entries) — those are addressed by `(section_type, major_start)`
    /// instead.
    pub fn get(&self, name: &str) -> Option<&CatalogViewEntry> {
        self.entries
            .iter()
            .find(|e| e.name.as_deref() == Some(name))
    }

    /// Filter + sort helper for reader construction. Returns borrowed
    /// references to every `entries[i]` matching the predicate, sorted
    /// by `stats.major_start` (entries without stats sink to the end).
    /// Mirrors `FullCatalog::shards_sorted` semantics — the
    /// `BackedCsrReader::new*` constructors use this to build their
    /// lightweight per-shard table in a single pass over the view.
    fn shards_filter_sorted<F: FnMut(&CatalogViewEntry) -> bool>(
        &self,
        mut keep: F,
    ) -> Vec<&CatalogViewEntry> {
        let mut shards: Vec<&CatalogViewEntry> = self.entries.iter().filter(|e| keep(e)).collect();
        shards.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.major_start));
        shards
    }

    /// CSR shard entries (`SectionType::CsrShard`) sorted by row range.
    /// Used by `BackedCsrReader::new` to build the X-shard table.
    pub fn csr_shards_sorted(&self) -> Vec<&CatalogViewEntry> {
        self.shards_filter_sorted(|e| e.section_type == SectionType::CsrShard)
    }

    /// CSR shard entries belonging to `modality_id`, sorted by row
    /// range. Used by `BackedCsrReader::for_modality`. `modality_id = 0`
    /// returns the global / single-modality shards on v1 and on
    /// single-modality v2 files.
    pub fn csr_shards_for_modality(&self, modality_id: u8) -> Vec<&CatalogViewEntry> {
        self.shards_filter_sorted(|e| {
            e.section_type == SectionType::CsrShard && e.modality_id == modality_id
        })
    }

    /// Layer-CSR shard entries whose retained name starts with
    /// `name_prefix`, sorted by row range. Used by
    /// `BackedCsrReader::new_for_layer`. Pure CSR shard entries dropped
    /// their name in the view; layer entries retain it specifically so
    /// this prefix filter still works.
    pub fn layer_csr_shards_sorted_with_prefix(&self, name_prefix: &str) -> Vec<&CatalogViewEntry> {
        self.shards_filter_sorted(|e| {
            e.section_type == SectionType::LayerCsrShard
                && e.name
                    .as_deref()
                    .is_some_and(|n| n.starts_with(name_prefix))
        })
    }
}

/// Compile-time guarantee that `CatalogView` is safe to share across
/// threads via `Arc<CatalogView>`. The constraint is structural: all
/// fields are `Send + Sync` (primitives, `Vec`, `Box<str>`), and the
/// type exposes no interior mutability.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CatalogView>();
    assert_send_sync::<Arc<CatalogView>>();
};

#[cfg(test)]
mod tests {
    use super::*;

    use byteorder::WriteBytesExt;

    use crate::catalog::{
        column_name_hash, ColumnStat, FullCatalogEntry, ShardStats, CURRENT_CATALOG_VERSION,
        SHARD_STATS_BASE_SIZE_V1,
    };

    fn sample_csr_stats(row_start: u64, row_end: u64, nnz: u64) -> ShardStats {
        ShardStats {
            row_start,
            row_end,
            col_start: 0,
            col_end: 30_000,
            nnz,
            value_min: 1,
            value_max: 65535,
            value_sum: 50_000_000,
            n_indexed_columns: 0,
            column_stats: vec![],
        }
    }

    fn sample_csc_stats_v2(col_start: u64, col_end: u64, nnz: u64) -> ShardStats {
        // v2 CSC: row pair carries [0, n_obs), col pair carries the
        // meaningful column range.
        ShardStats {
            row_start: 0,
            row_end: 1_000,
            col_start,
            col_end,
            nnz,
            value_min: 0,
            value_max: 0,
            value_sum: 0,
            n_indexed_columns: 0,
            column_stats: vec![],
        }
    }

    fn build_full(entries: Vec<FullCatalogEntry>, n_obs: u64) -> FullCatalog {
        FullCatalog {
            catalog_version: CURRENT_CATALOG_VERSION,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            n_obs,
            entries,
            data_generation: 0,
            csc_build_generation: 0,
        }
    }

    /// Field-for-field equivalence between a `CatalogView` parsed
    /// directly from bytes and one derived from `FullCatalog`. This is
    /// the load-bearing correctness check for the lightweight path —
    /// substituting the view in reader construction must not change
    /// any read-relevant metadata.
    #[test]
    fn read_from_bytes_matches_from_full() {
        let entries = vec![
            FullCatalogEntry {
                name: "obs".into(),
                offset: 4352,
                length: 4096,
                section_type: SectionType::ObsMetadata,
                checksum: [0xAA; 32],
                modality_id: 0,
                stats: None,
            },
            FullCatalogEntry {
                name: "var".into(),
                offset: 8448,
                length: 4096,
                section_type: SectionType::VarMetadata,
                checksum: [0xBB; 32],
                modality_id: 0,
                stats: None,
            },
            FullCatalogEntry {
                name: "X_shard_0".into(),
                offset: 12_544,
                length: 200_000,
                section_type: SectionType::CsrShard,
                checksum: [0xCC; 32],
                modality_id: 0,
                stats: Some(sample_csr_stats(0, 16_384, 3_276_800)),
            },
            FullCatalogEntry {
                name: "X_shard_1".into(),
                offset: 212_544,
                length: 200_000,
                section_type: SectionType::CsrShard,
                checksum: [0xDD; 32],
                modality_id: 0,
                stats: Some(sample_csr_stats(16_384, 32_768, 3_500_000)),
            },
            FullCatalogEntry {
                name: "layer/raw/shard_0".into(),
                offset: 412_544,
                length: 100_000,
                section_type: SectionType::LayerCsrShard,
                checksum: [0xEE; 32],
                modality_id: 0,
                stats: Some(sample_csr_stats(0, 16_384, 1_000_000)),
            },
            FullCatalogEntry {
                name: "X_csc_0".into(),
                offset: 512_544,
                length: 50_000,
                section_type: SectionType::CscShard,
                checksum: [0xFF; 32],
                modality_id: 0,
                stats: Some(sample_csc_stats_v2(0, 15_000, 500_000)),
            },
        ];
        let full = build_full(entries, 32_768);

        let mut buf = Vec::new();
        full.write_to(&mut buf).unwrap();

        let from_bytes = CatalogView::read_from_bytes(&buf, true).unwrap();
        let from_full = CatalogView::from_full(&full);

        assert_eq!(from_bytes.catalog_version, from_full.catalog_version);
        assert_eq!(from_bytes.manifest_sequence, from_full.manifest_sequence);
        assert_eq!(
            from_bytes.prev_catalog_offset,
            from_full.prev_catalog_offset
        );
        assert_eq!(from_bytes.n_obs, from_full.n_obs);
        assert_eq!(from_bytes.entries.len(), from_full.entries.len());

        for (a, b) in from_bytes.entries.iter().zip(from_full.entries.iter()) {
            assert_eq!(a.name, b.name, "name mismatch");
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.length, b.length);
            assert_eq!(a.section_type, b.section_type);
            assert_eq!(a.modality_id, b.modality_id);
            assert_eq!(a.stats, b.stats, "stats mismatch for {:?}", a.section_type);
        }
    }

    /// Pure X-shard entries drop their `name` in the view — that's the
    /// allocation win. Layer + non-shard entries retain it.
    #[test]
    fn x_shard_names_dropped_layer_names_kept() {
        let entries = vec![
            FullCatalogEntry {
                name: "X_shard_0".into(),
                offset: 4352,
                length: 1000,
                section_type: SectionType::CsrShard,
                checksum: [0u8; 32],
                modality_id: 0,
                stats: Some(sample_csr_stats(0, 100, 1000)),
            },
            FullCatalogEntry {
                name: "layer/normalized/shard_0".into(),
                offset: 5352,
                length: 1000,
                section_type: SectionType::LayerCsrShard,
                checksum: [0u8; 32],
                modality_id: 0,
                stats: Some(sample_csr_stats(0, 100, 1000)),
            },
            FullCatalogEntry {
                name: "obs".into(),
                offset: 6352,
                length: 1000,
                section_type: SectionType::ObsMetadata,
                checksum: [0u8; 32],
                modality_id: 0,
                stats: None,
            },
        ];
        let full = build_full(entries, 100);
        let mut buf = Vec::new();
        full.write_to(&mut buf).unwrap();
        let view = CatalogView::read_from_bytes(&buf, true).unwrap();

        assert_eq!(view.entries[0].section_type, SectionType::CsrShard);
        assert!(
            view.entries[0].name.is_none(),
            "X-shard name should be dropped, got: {:?}",
            view.entries[0].name
        );

        assert_eq!(view.entries[1].section_type, SectionType::LayerCsrShard);
        assert_eq!(
            view.entries[1].name(),
            Some("layer/normalized/shard_0"),
            "layer name must be retained for prefix filtering"
        );

        assert_eq!(view.entries[2].section_type, SectionType::ObsMetadata);
        assert_eq!(
            view.entries[2].name(),
            Some("obs"),
            "non-shard name must be retained for `get()` lookup"
        );

        // `get()` is name-based — works for layers and metadata but
        // not pure shards, by design.
        assert!(view.get("obs").is_some());
        assert!(view.get("X_shard_0").is_none());
    }

    /// v1 files read through the v2 reader path. The lightweight
    /// parser must (a) accept the 53-byte-per-entry v1 layout (no
    /// `modality_id` byte) and (b) interpret v1 stats correctly —
    /// for CSC shards the axis-overloaded row pair carries the
    /// column range, which the lite view exposes as the major axis.
    #[test]
    fn v1_csc_axis_overload_resolves_to_major_range() {
        // Hand-build a v1 catalog with one CSR shard and one CSC
        // shard. Same payload shape as the in-tree
        // `v1_catalog_csc_axis_reconciliation` regression test, but
        // we drive the lightweight parser through it.
        let mut payload = Vec::new();
        payload.write_u16::<LittleEndian>(1).unwrap(); // v1
        payload.write_u64::<LittleEndian>(0).unwrap(); // manifest_seq
        payload.write_u64::<LittleEndian>(0).unwrap(); // prev_offset
        payload.write_u64::<LittleEndian>(1000).unwrap(); // n_obs
        payload.write_u32::<LittleEndian>(2).unwrap(); // n_entries

        // Entry 0: CSR shard, rows [0, 100), nnz=1000
        let mut v1_csr_stats = Vec::new();
        v1_csr_stats.write_u64::<LittleEndian>(0).unwrap();
        v1_csr_stats.write_u64::<LittleEndian>(100).unwrap();
        v1_csr_stats.write_u64::<LittleEndian>(1000).unwrap();
        v1_csr_stats.write_u32::<LittleEndian>(1).unwrap();
        v1_csr_stats.write_u32::<LittleEndian>(255).unwrap();
        v1_csr_stats.write_u64::<LittleEndian>(50_000).unwrap();
        v1_csr_stats.push(0);
        assert_eq!(v1_csr_stats.len(), SHARD_STATS_BASE_SIZE_V1);

        let name = b"X_shard_0";
        payload
            .write_u16::<LittleEndian>(name.len() as u16)
            .unwrap();
        payload.extend_from_slice(name);
        payload.write_u64::<LittleEndian>(4352).unwrap();
        payload.write_u64::<LittleEndian>(1000).unwrap();
        payload.push(SectionType::CsrShard as u8);
        payload.extend_from_slice(&[0u8; 32]);
        // no modality_id (v1)
        payload
            .write_u16::<LittleEndian>(v1_csr_stats.len() as u16)
            .unwrap();
        payload.extend_from_slice(&v1_csr_stats);

        // Entry 1: CSC shard. v1 axis-overload — the row_* pair on disk
        // is really the col range [200, 350).
        let mut v1_csc_stats = Vec::new();
        v1_csc_stats.write_u64::<LittleEndian>(200).unwrap();
        v1_csc_stats.write_u64::<LittleEndian>(350).unwrap();
        v1_csc_stats.write_u64::<LittleEndian>(5000).unwrap();
        v1_csc_stats.write_u32::<LittleEndian>(0).unwrap();
        v1_csc_stats.write_u32::<LittleEndian>(0).unwrap();
        v1_csc_stats.write_u64::<LittleEndian>(0).unwrap();
        v1_csc_stats.push(0);

        let csc_name = b"X_csc_0";
        payload
            .write_u16::<LittleEndian>(csc_name.len() as u16)
            .unwrap();
        payload.extend_from_slice(csc_name);
        payload.write_u64::<LittleEndian>(8000).unwrap();
        payload.write_u64::<LittleEndian>(2000).unwrap();
        payload.push(SectionType::CscShard as u8);
        payload.extend_from_slice(&[0u8; 32]);
        payload
            .write_u16::<LittleEndian>(v1_csc_stats.len() as u16)
            .unwrap();
        payload.extend_from_slice(&v1_csc_stats);

        let checksum = crate::checksum::blake3_hash(&payload);
        let mut full = payload.clone();
        full.extend_from_slice(&checksum);

        let view = CatalogView::read_from_bytes(&full, true).unwrap();
        assert_eq!(view.catalog_version, 1);
        assert_eq!(view.entries.len(), 2);

        let csr = &view.entries[0];
        assert_eq!(csr.section_type, SectionType::CsrShard);
        assert_eq!(csr.modality_id, 0, "v1 should stamp modality_id=0");
        let csr_stats = csr.stats.as_ref().unwrap();
        assert_eq!(csr_stats.major_start, 0);
        assert_eq!(csr_stats.major_end, 100);
        assert_eq!(csr_stats.nnz, 1000);

        let csc = &view.entries[1];
        assert_eq!(csc.section_type, SectionType::CscShard);
        let csc_stats = csc.stats.as_ref().unwrap();
        // v1 CSC: the axis-overloaded row_* pair IS the column range.
        // The lightweight view returns it as the major axis directly
        // — no FullCatalog-style reconcile pass required.
        assert_eq!(csc_stats.major_start, 200);
        assert_eq!(csc_stats.major_end, 350);
        assert_eq!(csc_stats.nnz, 5000);
    }

    /// Multimodal v2 catalog with three modalities. The lightweight
    /// view must round-trip `modality_id` per entry so the
    /// `BackedCsrReader::for_modality` constructor works against it.
    #[test]
    fn v2_multimodal_modality_ids_preserved() {
        let entries = vec![
            FullCatalogEntry {
                name: "obs".into(),
                offset: 4352,
                length: 1000,
                section_type: SectionType::ObsMetadata,
                checksum: [0u8; 32],
                modality_id: 0,
                stats: None,
            },
            // Three CSR shards, one per modality.
            FullCatalogEntry {
                name: "X_shard_0".into(),
                offset: 5352,
                length: 1000,
                section_type: SectionType::CsrShard,
                checksum: [0u8; 32],
                modality_id: 0,
                stats: Some(sample_csr_stats(0, 100, 1000)),
            },
            FullCatalogEntry {
                name: "X_shard_0_adt".into(),
                offset: 6352,
                length: 1000,
                section_type: SectionType::CsrShard,
                checksum: [0u8; 32],
                modality_id: 1,
                stats: Some(sample_csr_stats(0, 100, 1000)),
            },
            FullCatalogEntry {
                name: "X_shard_0_atac".into(),
                offset: 7352,
                length: 1000,
                section_type: SectionType::CsrShard,
                checksum: [0u8; 32],
                modality_id: 2,
                stats: Some(sample_csr_stats(0, 100, 1000)),
            },
        ];
        let full = build_full(entries, 100);
        let mut buf = Vec::new();
        full.write_to(&mut buf).unwrap();
        let view = CatalogView::read_from_bytes(&buf, true).unwrap();

        let modalities: Vec<u8> = view
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .map(|e| e.modality_id)
            .collect();
        assert_eq!(modalities, vec![0, 1, 2]);
    }

    /// Files with predicate-pushdown column stats: the lightweight
    /// view drops `column_stats` entirely. `stats_len` reports the
    /// inflated payload size, but the lite parser only consumes the
    /// 24 bytes it needs (row_start / row_end / nnz) and skips the
    /// rest via the `stats_len` advance. Confirms there is no
    /// regression when the catalog carries column statistics.
    #[test]
    fn column_stats_payload_is_skipped_not_parsed() {
        let stats = ShardStats {
            row_start: 0,
            row_end: 256,
            col_start: 0,
            col_end: 1000,
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
                    bitset: vec![0xFF, 0x0F],
                },
            ],
        };
        let entries = vec![FullCatalogEntry {
            name: "X_shard_0".into(),
            offset: 4352,
            length: 1000,
            section_type: SectionType::CsrShard,
            checksum: [0u8; 32],
            modality_id: 0,
            stats: Some(stats),
        }];
        let full = build_full(entries, 256);
        let mut buf = Vec::new();
        full.write_to(&mut buf).unwrap();
        let view = CatalogView::read_from_bytes(&buf, true).unwrap();

        let s = view.entries[0].stats.as_ref().unwrap();
        assert_eq!(s.major_start, 0);
        assert_eq!(s.major_end, 256);
        assert_eq!(s.nnz, 50_000);
    }

    /// Checksum semantics must match `FullCatalog::read_from`:
    /// `verify_checksum=true` rejects corrupted payloads,
    /// `verify_checksum=false` accepts them.
    #[test]
    fn checksum_verification_round_trips() {
        let entries = vec![FullCatalogEntry {
            name: "X_shard_0".into(),
            offset: 4352,
            length: 1000,
            section_type: SectionType::CsrShard,
            checksum: [0u8; 32],
            modality_id: 0,
            stats: Some(sample_csr_stats(0, 100, 1000)),
        }];
        let full = build_full(entries, 100);
        let mut buf = Vec::new();
        full.write_to(&mut buf).unwrap();

        // Clean read with verify.
        CatalogView::read_from_bytes(&buf, true).unwrap();

        // Flip a payload byte → ChecksumMismatch (matches FullCatalog).
        let mut corrupt = buf.clone();
        corrupt[10] ^= 0xFF;
        let err = CatalogView::read_from_bytes(&corrupt, true).unwrap_err();
        assert!(matches!(err, ScxError::ChecksumMismatch { .. }));

        // Same corrupt bytes with verify=false should parse fine.
        CatalogView::read_from_bytes(&corrupt, false).unwrap();
    }
}
