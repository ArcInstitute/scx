// RootCatalog + FullCatalog (SPEC §3.2)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::checksum::blake3_hash;
use crate::error::{Result, ScxError};
use crate::section::SectionType;

/// Maximum size of the serialized root catalog in bytes.
pub const ROOT_CATALOG_MAX_SIZE: usize = 4096;

/// Size of a single root catalog entry: 1 + 8 + 8 + 4 + 32 = 53 bytes.
pub const ROOT_CATALOG_ENTRY_SIZE: usize = 53;

// ---------------------------------------------------------------------------
// RootCatalog
// ---------------------------------------------------------------------------

/// A single entry in the root catalog, representing a section group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootCatalogEntry {
    /// Section type ID for the group.
    pub group_type: u8,
    /// File offset of the first section in this group.
    pub first_section_offset: u64,
    /// Total byte length of all sections in this group.
    pub total_group_length: u64,
    /// Number of sections in this group.
    pub n_sections: u32,
    /// 32-byte summary (application-defined).
    pub summary: [u8; 32],
}

impl RootCatalogEntry {
    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u8(self.group_type)?;
        w.write_u64::<LittleEndian>(self.first_section_offset)?;
        w.write_u64::<LittleEndian>(self.total_group_length)?;
        w.write_u32::<LittleEndian>(self.n_sections)?;
        w.write_all(&self.summary)?;
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let group_type = r.read_u8()?;
        let first_section_offset = r.read_u64::<LittleEndian>()?;
        let total_group_length = r.read_u64::<LittleEndian>()?;
        let n_sections = r.read_u32::<LittleEndian>()?;
        let mut summary = [0u8; 32];
        r.read_exact(&mut summary)?;
        Ok(Self {
            group_type,
            first_section_offset,
            total_group_length,
            n_sections,
            summary,
        })
    }
}

/// The compact root catalog stored at offset 256, max 4096 bytes.
/// Provides a quick summary of section groups for fast file open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootCatalog {
    pub n_section_groups: u16,
    pub entries: Vec<RootCatalogEntry>,
}

impl RootCatalog {
    /// Serialize the root catalog. Returns `RootCatalogTooLarge` if > 4096 bytes.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut buf = Vec::new();
        buf.write_u16::<LittleEndian>(self.n_section_groups)?;
        for entry in &self.entries {
            entry.write_to(&mut buf)?;
        }
        if buf.len() > ROOT_CATALOG_MAX_SIZE {
            return Err(ScxError::RootCatalogTooLarge(buf.len()));
        }
        w.write_all(&buf)?;
        Ok(())
    }

    /// Deserialize a root catalog from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let n_section_groups = r.read_u16::<LittleEndian>()?;
        let mut entries = Vec::with_capacity(n_section_groups as usize);
        for _ in 0..n_section_groups {
            entries.push(RootCatalogEntry::read_from(r)?);
        }
        Ok(Self {
            n_section_groups,
            entries,
        })
    }
}

// ---------------------------------------------------------------------------
// ColumnStat — per-column statistics for shard pruning (SPEC §3.2, Phase 2)
// ---------------------------------------------------------------------------

/// Per-column statistic stored in ShardStats for predicate pushdown.
///
/// - `MinMax`: stat_type == 0 in SPEC §3.2. Stores numeric min/max.
/// - `CategoryBitset`: stat_type == 1 in SPEC §3.2. Bit i set if dictionary index i is present.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnStat {
    /// stat_type == 0 in SPEC §3.2
    MinMax {
        column_name_hash: u64, // BLAKE3 truncated hash of column name
        min: f64,
        max: f64,
    },
    /// stat_type == 1 in SPEC §3.2
    CategoryBitset {
        column_name_hash: u64,
        bitset: Vec<u8>, // bit i set if dictionary index i present
    },
}

impl ColumnStat {
    /// Return the column name hash for this stat.
    pub fn column_name_hash(&self) -> u64 {
        match self {
            ColumnStat::MinMax {
                column_name_hash, ..
            } => *column_name_hash,
            ColumnStat::CategoryBitset {
                column_name_hash, ..
            } => *column_name_hash,
        }
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        match self {
            ColumnStat::MinMax {
                column_name_hash,
                min,
                max,
            } => {
                w.write_u8(0)?; // stat_type
                w.write_u64::<LittleEndian>(*column_name_hash)?;
                w.write_f64::<LittleEndian>(*min)?;
                w.write_f64::<LittleEndian>(*max)?;
            }
            ColumnStat::CategoryBitset {
                column_name_hash,
                bitset,
            } => {
                w.write_u8(1)?; // stat_type
                w.write_u64::<LittleEndian>(*column_name_hash)?;
                w.write_u16::<LittleEndian>(bitset.len() as u16)?;
                w.write_all(bitset)?;
            }
        }
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let stat_type = r.read_u8()?;
        match stat_type {
            0 => {
                let column_name_hash = r.read_u64::<LittleEndian>()?;
                let min = r.read_f64::<LittleEndian>()?;
                let max = r.read_f64::<LittleEndian>()?;
                Ok(ColumnStat::MinMax {
                    column_name_hash,
                    min,
                    max,
                })
            }
            1 => {
                let column_name_hash = r.read_u64::<LittleEndian>()?;
                let bitset_len = r.read_u16::<LittleEndian>()? as usize;
                let mut bitset = vec![0u8; bitset_len];
                r.read_exact(&mut bitset)?;
                Ok(ColumnStat::CategoryBitset {
                    column_name_hash,
                    bitset,
                })
            }
            _ => Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown column stat type: {stat_type}"),
            ))),
        }
    }
}

/// Compute a BLAKE3 truncated-64 hash of a column name.
/// Used as the key for looking up column stats during pushdown.
pub fn column_name_hash(name: &str) -> u64 {
    let bytes = crate::checksum::blake3_truncated_64(name.as_bytes());
    u64::from_le_bytes(bytes)
}

// ---------------------------------------------------------------------------
// ShardStats
// ---------------------------------------------------------------------------

/// Per-shard statistics stored in FullCatalogEntry.
/// Phase 1: n_indexed_columns is always 0, column_stats is empty.
/// Phase 2: column_stats may contain per-column MinMax or CategoryBitset entries.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardStats {
    pub row_start: u64,
    pub row_end: u64,
    pub nnz: u64,
    pub value_min: u32,
    pub value_max: u32,
    pub value_sum: u64,
    pub n_indexed_columns: u8,
    /// Per-column statistics for predicate pushdown (Phase 2).
    /// Length must equal `n_indexed_columns`.
    pub column_stats: Vec<ColumnStat>,
}

/// Serialized size of ShardStats when n_indexed_columns == 0.
pub const SHARD_STATS_BASE_SIZE: usize = 41; // 8+8+8+4+4+8+1

impl ShardStats {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u64::<LittleEndian>(self.row_start)?;
        w.write_u64::<LittleEndian>(self.row_end)?;
        w.write_u64::<LittleEndian>(self.nnz)?;
        w.write_u32::<LittleEndian>(self.value_min)?;
        w.write_u32::<LittleEndian>(self.value_max)?;
        w.write_u64::<LittleEndian>(self.value_sum)?;
        w.write_u8(self.column_stats.len() as u8)?;
        for cs in &self.column_stats {
            cs.write_to(w)?;
        }
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let row_start = r.read_u64::<LittleEndian>()?;
        let row_end = r.read_u64::<LittleEndian>()?;
        let nnz = r.read_u64::<LittleEndian>()?;
        let value_min = r.read_u32::<LittleEndian>()?;
        let value_max = r.read_u32::<LittleEndian>()?;
        let value_sum = r.read_u64::<LittleEndian>()?;
        let n_indexed_columns = r.read_u8()?;
        let mut column_stats = Vec::with_capacity(n_indexed_columns as usize);
        for _ in 0..n_indexed_columns {
            column_stats.push(ColumnStat::read_from(r)?);
        }
        Ok(Self {
            row_start,
            row_end,
            nnz,
            value_min,
            value_max,
            value_sum,
            n_indexed_columns,
            column_stats,
        })
    }
}

// ---------------------------------------------------------------------------
// FullCatalog
// ---------------------------------------------------------------------------

/// A single entry in the full catalog, indexing one section in the file.
#[derive(Debug, Clone)]
pub struct FullCatalogEntry {
    /// Section name (e.g. "X_shard_0", "obs", "var").
    pub name: String,
    /// File offset of this section.
    pub offset: u64,
    /// Byte length of this section.
    pub length: u64,
    /// Section type.
    pub section_type: SectionType,
    /// BLAKE3 checksum of the section data.
    pub checksum: [u8; 32],
    /// Optional shard statistics (present for CSR/CSC shard entries).
    pub stats: Option<ShardStats>,
}

/// The full catalog stored at the end of the file, indexing every section
/// with BLAKE3 checksums.
#[derive(Debug, Clone)]
pub struct FullCatalog {
    pub catalog_version: u16,
    pub manifest_sequence: u64,
    pub prev_catalog_offset: u64,
    pub n_obs: u64,
    pub entries: Vec<FullCatalogEntry>,
}

impl FullCatalog {
    /// Serialize the full catalog. Appends a trailing 32-byte BLAKE3 checksum.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut buf = Vec::new();

        // Header fields
        buf.write_u16::<LittleEndian>(self.catalog_version)?;
        buf.write_u64::<LittleEndian>(self.manifest_sequence)?;
        buf.write_u64::<LittleEndian>(self.prev_catalog_offset)?;
        buf.write_u64::<LittleEndian>(self.n_obs)?;
        buf.write_u32::<LittleEndian>(self.entries.len() as u32)?;

        // Entries
        for entry in &self.entries {
            let name_bytes = entry.name.as_bytes();
            buf.write_u16::<LittleEndian>(name_bytes.len() as u16)?;
            buf.write_all(name_bytes)?;
            buf.write_u64::<LittleEndian>(entry.offset)?;
            buf.write_u64::<LittleEndian>(entry.length)?;
            buf.write_u8(entry.section_type as u8)?;
            buf.write_all(&entry.checksum)?;

            // Stats: write length prefix then optional stats bytes
            match &entry.stats {
                Some(stats) => {
                    let mut stats_buf = Vec::new();
                    stats.write_to(&mut stats_buf)?;
                    buf.write_u16::<LittleEndian>(stats_buf.len() as u16)?;
                    buf.write_all(&stats_buf)?;
                }
                None => {
                    buf.write_u16::<LittleEndian>(0)?;
                }
            }
        }

        // Compute checksum of everything above and append
        let checksum = blake3_hash(&buf);
        w.write_all(&buf)?;
        w.write_all(&checksum)?;
        Ok(())
    }

    /// Deserialize a full catalog. `total_len` is the total byte count
    /// (including the trailing 32-byte checksum) as indicated by the file header.
    ///
    /// When `verify_checksum` is `true`, the trailing BLAKE3 checksum is
    /// validated before parsing. When `false`, the checksum bytes are still
    /// consumed but not verified, which is useful for trusted-source reads
    /// where catalog integrity is assumed.
    pub fn read_from<R: Read>(r: &mut R, total_len: usize, verify_checksum: bool) -> Result<Self> {
        if total_len < 32 {
            return Err(ScxError::ChecksumMismatch);
        }

        let mut all_bytes = vec![0u8; total_len];
        r.read_exact(&mut all_bytes)?;

        let payload_len = total_len - 32;
        let payload = &all_bytes[..payload_len];

        if verify_checksum {
            let expected_checksum = &all_bytes[payload_len..];
            let computed = blake3_hash(payload);
            if computed[..] != *expected_checksum {
                return Err(ScxError::ChecksumMismatch);
            }
        }

        // Parse the payload
        let mut cur = std::io::Cursor::new(payload);
        let catalog_version = cur.read_u16::<LittleEndian>()?;
        let manifest_sequence = cur.read_u64::<LittleEndian>()?;
        let prev_catalog_offset = cur.read_u64::<LittleEndian>()?;
        let n_obs = cur.read_u64::<LittleEndian>()?;
        let n_entries = cur.read_u32::<LittleEndian>()? as usize;

        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            let name_len = cur.read_u16::<LittleEndian>()? as usize;
            let mut name_bytes = vec![0u8; name_len];
            cur.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let offset = cur.read_u64::<LittleEndian>()?;
            let length = cur.read_u64::<LittleEndian>()?;
            let section_type_raw = cur.read_u8()?;

            let mut checksum = [0u8; 32];
            cur.read_exact(&mut checksum)?;

            let stats_len = cur.read_u16::<LittleEndian>()? as usize;
            let stats = if stats_len > 0 {
                let mut stats_bytes = vec![0u8; stats_len];
                cur.read_exact(&mut stats_bytes)?;
                let mut stats_cur = std::io::Cursor::new(&stats_bytes);
                Some(ShardStats::read_from(&mut stats_cur)?)
            } else {
                None
            };

            // Skip unknown section types (forward-compatibility)
            let section_type = match SectionType::from_u8(section_type_raw) {
                Some(st) => st,
                None => continue,
            };

            entries.push(FullCatalogEntry {
                name,
                offset,
                length,
                section_type,
                checksum,
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

    /// Look up a catalog entry by name.
    pub fn get(&self, name: &str) -> Option<&FullCatalogEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Return all entries matching a given section type.
    pub fn shards(&self, section_type: SectionType) -> Vec<&FullCatalogEntry> {
        self.entries
            .iter()
            .filter(|e| e.section_type == section_type)
            .collect()
    }

    /// Return CSR shard entries sorted by `stats.row_start`.
    /// Entries without stats are placed at the end.
    pub fn shards_sorted(&self) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .collect();
        shards.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));
        shards
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(buf.len(), SHARD_STATS_BASE_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = ShardStats::read_from(&mut cursor).unwrap();
        assert_eq!(decoded, stats);
    }

    #[test]
    fn shard_stats_extreme_values() {
        let stats = ShardStats {
            row_start: u64::MAX - 1,
            row_end: u64::MAX,
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
        let decoded = ShardStats::read_from(&mut cursor).unwrap();
        assert_eq!(decoded, stats);
    }

    #[test]
    fn shard_stats_with_column_stats_round_trip() {
        let stats = ShardStats {
            row_start: 0,
            row_end: 1000,
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
        assert!(buf.len() > SHARD_STATS_BASE_SIZE); // larger than base

        let mut cursor = Cursor::new(&buf);
        let decoded = ShardStats::read_from(&mut cursor).unwrap();
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
            nnz: 200,
            value_min: 1,
            value_max: 10,
            value_sum: 500,
            n_indexed_columns: 0,
            column_stats: vec![],
        };
        let mut buf = Vec::new();
        stats.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), SHARD_STATS_BASE_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = ShardStats::read_from(&mut cursor).unwrap();
        assert_eq!(decoded, stats);
        assert!(decoded.column_stats.is_empty());
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
            stats: if with_stats {
                Some(sample_stats())
            } else {
                None
            },
        }
    }

    fn sample_full_catalog() -> FullCatalog {
        FullCatalog {
            catalog_version: 1,
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
        assert!(matches!(err, ScxError::ChecksumMismatch));
    }

    /// 9.15: Empty catalog (0 entries) round-trip
    #[test]
    fn full_catalog_empty() {
        let catalog = FullCatalog {
            catalog_version: 1,
            manifest_sequence: 0,
            prev_catalog_offset: 0,
            n_obs: 0,
            entries: vec![],
        };

        let mut buf = Vec::new();
        catalog.write_to(&mut buf).unwrap();

        let total_len = buf.len();
        let mut cursor = Cursor::new(&buf);
        let decoded = FullCatalog::read_from(&mut cursor, total_len, true).unwrap();

        assert_eq!(decoded.catalog_version, 1);
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
}
