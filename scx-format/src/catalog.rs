// RootCatalog + FullCatalog (docs/format.md (Dual Catalog))

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
// ColumnStat — per-column statistics for shard pruning (docs/format.md (Dual Catalog), Phase 2)
// ---------------------------------------------------------------------------

/// Per-column statistic stored in ShardStats for predicate pushdown.
///
/// - `MinMax`: stat_type == 0 in docs/format.md (Dual Catalog). Stores numeric min/max.
/// - `CategoryBitset`: stat_type == 1 in docs/format.md (Dual Catalog). Bit i set if dictionary index i is present.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnStat {
    /// stat_type == 0 in docs/format.md (Dual Catalog)
    MinMax {
        column_name_hash: u64, // BLAKE3 truncated hash of column name
        min: f64,
        max: f64,
    },
    /// stat_type == 1 in docs/format.md (Dual Catalog)
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
///
/// v1 layout: `row_start`/`row_end` only. For `CscShard` entries the
/// row pair was axis-overloaded — used as `col_start`/`col_end`. v2
/// adds explicit `col_start`/`col_end` fields written unconditionally
/// for symmetry: a CSR shard carries `[0, n_vars)` in the column pair
/// (full column range), and a CSC shard carries `[0, n_obs)` in the
/// row pair (full row range). On v1 catalog reads the CSC entries are
/// reconciled by `FullCatalog::read_from` (the row pair is copied into
/// the column pair), so consumers can always use the `col_range()`
/// accessor on CSC shards regardless of catalog version.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardStats {
    /// Row-axis start. For CSR/Layer/Obsp shards: the global row index
    /// where this shard begins. For v2 CSC shards: 0 (full row range).
    /// For v1 CSC shards before reconciliation: axis-overloaded
    /// col_start (legacy on-disk encoding).
    pub row_start: u64,
    /// Row-axis end. For CSR/Layer/Obsp: row_start + n_rows. For v2
    /// CSC: n_obs. For v1 CSC before reconciliation: col_end.
    pub row_end: u64,
    /// Column-axis start. For CSR/Layer/Obsp v2: 0. For CSC v2: the
    /// global column index where this shard begins. v2-only field —
    /// zero on v1 reads unless reconciled by `FullCatalog::read_from`.
    pub col_start: u64,
    /// Column-axis end. For CSR/Layer/Obsp v2: n_vars. For CSC v2:
    /// col_start + n_cols. v2-only field — zero on v1 reads unless
    /// reconciled.
    pub col_end: u64,
    pub nnz: u64,
    pub value_min: u32,
    pub value_max: u32,
    pub value_sum: u64,
    pub n_indexed_columns: u8,
    /// Per-column statistics for predicate pushdown (Phase 2).
    /// Length must equal `n_indexed_columns`.
    pub column_stats: Vec<ColumnStat>,
}

/// Serialized size of v1 ShardStats when n_indexed_columns == 0.
/// v1 layout: row_start(8) + row_end(8) + nnz(8) + value_min(4) +
/// value_max(4) + value_sum(8) + n_indexed_columns(1) = 41 bytes.
pub const SHARD_STATS_BASE_SIZE_V1: usize = 41;

/// Serialized size of v2 ShardStats when n_indexed_columns == 0.
/// v2 layout: v1 + col_start(8) + col_end(8) = 57 bytes.
pub const SHARD_STATS_BASE_SIZE_V2: usize = 57;

/// Backward-compat alias for the v1 base size. New code should prefer
/// the explicit `_V1` / `_V2` constants.
#[deprecated(note = "Use SHARD_STATS_BASE_SIZE_V1 or SHARD_STATS_BASE_SIZE_V2")]
pub const SHARD_STATS_BASE_SIZE: usize = SHARD_STATS_BASE_SIZE_V1;

impl ShardStats {
    /// Generic major-axis start: `row_start` for CSR/Layer/Obsp,
    /// `col_start` for CSC. The accessor dispatches on `section_type`
    /// because v2 CSC stats no longer axis-overload the row pair —
    /// the row pair carries `[0, n_obs)` (full row range), while the
    /// meaningful column range lives in `col_start`/`col_end`. For v1
    /// CSC entries reconciled at catalog-read time the two pairs are
    /// equal, so this accessor returns the same value either way.
    pub fn major_start(&self, section_type: SectionType) -> u64 {
        match section_type {
            SectionType::CscShard => self.col_start,
            _ => self.row_start,
        }
    }

    /// Generic major-axis end: `row_end` for CSR/Layer/Obsp,
    /// `col_end` for CSC. See [`Self::major_start`] for the rationale.
    pub fn major_end(&self, section_type: SectionType) -> u64 {
        match section_type {
            SectionType::CscShard => self.col_end,
            _ => self.row_end,
        }
    }

    /// Row range. Pick this method when the entry is known to be a
    /// row-major shard (CSR / LayerCsr / ObspCsr).
    pub fn row_range(&self) -> std::ops::Range<u64> {
        self.row_start..self.row_end
    }

    /// Column range. Pick this method when the entry is known to be a
    /// `CscShard`. v2 stats carry an explicit column range; v1 stats
    /// reconciled at catalog-read time have `col_start`/`col_end`
    /// populated identically.
    pub fn col_range(&self) -> std::ops::Range<u64> {
        self.col_start..self.col_end
    }

    /// Serialize in v2 layout (always — writers emit v2 going forward).
    /// Field order: row_start, row_end, col_start, col_end, nnz,
    /// value_min, value_max, value_sum, n_indexed_columns, column_stats.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u64::<LittleEndian>(self.row_start)?;
        w.write_u64::<LittleEndian>(self.row_end)?;
        w.write_u64::<LittleEndian>(self.col_start)?;
        w.write_u64::<LittleEndian>(self.col_end)?;
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

    /// Deserialize, branching on `catalog_version`.
    ///
    /// - v1 (`catalog_version == 1`): reads the legacy 41-byte base
    ///   layout. `col_start` and `col_end` are left as 0; the parent
    ///   `FullCatalog::read_from` reconciles CSC entries afterwards by
    ///   copying `row_start`/`row_end` into the column pair.
    /// - v2 (`catalog_version >= 2`): reads the 57-byte base layout
    ///   including explicit `col_start`/`col_end`.
    pub fn read_from<R: Read>(r: &mut R, catalog_version: u16) -> Result<Self> {
        let row_start = r.read_u64::<LittleEndian>()?;
        let row_end = r.read_u64::<LittleEndian>()?;
        let (col_start, col_end) = if catalog_version >= 2 {
            let col_start = r.read_u64::<LittleEndian>()?;
            let col_end = r.read_u64::<LittleEndian>()?;
            (col_start, col_end)
        } else {
            (0u64, 0u64)
        };
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
            col_start,
            col_end,
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
    /// Modality routing key (Phase B). `0` is the global / primary
    /// modality (the implicit modality of every v1 file and the
    /// default for single-modality v2 files); `1..=n_modalities` are
    /// 1-based indices into the file's `ModalityTable`.
    ///
    /// On v1 catalog reads (no on-disk byte) this is always stamped
    /// `0`. On v2 reads it is parsed from the per-entry encoding.
    pub modality_id: u8,
    /// Optional shard statistics (present for CSR/CSC shard entries).
    pub stats: Option<ShardStats>,
}

/// The full catalog stored at the end of the file, indexing every section
/// with BLAKE3 checksums.
///
/// `catalog_version` is bumped from 1 to 2 alongside the file
/// `format_version` bump. v2 catalogs encode 16 extra bytes per entry
/// that carries stats (`col_start` + `col_end`); entries without stats
/// (`stats_len == 0`) pay nothing. v2 readers accept both v1 and v2
/// catalogs; v1 readers reject anything stamped `catalog_version >= 2`
/// via the file `format_version` check upstream.
pub const CURRENT_CATALOG_VERSION: u16 = 2;

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
    ///
    /// `ShardStats::write_to` always emits the v2 stats layout (with the
    /// explicit `col_start` / `col_end` pair). To keep the declared
    /// `catalog_version` consistent with what we actually write — readers
    /// branch on this field — we transparently upgrade `catalog_version` to
    /// at least 2 here. v2 is a strict superset of v1, so v1 consumers
    /// reading round-tripped catalogs still parse cleanly via the v2 path.
    /// This closes the symmetry break that broke `pyscx.pull` of any cloud
    /// `.scxd` directory whose source `.scx` was written before the v2 stats
    /// layout shipped (2026-05-10 tier-full gate run #2).
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let catalog_version = std::cmp::max(self.catalog_version, 2);

        let mut buf = Vec::new();

        // Header fields
        buf.write_u16::<LittleEndian>(catalog_version)?;
        buf.write_u64::<LittleEndian>(self.manifest_sequence)?;
        buf.write_u64::<LittleEndian>(self.prev_catalog_offset)?;
        buf.write_u64::<LittleEndian>(self.n_obs)?;
        buf.write_u32::<LittleEndian>(self.entries.len() as u32)?;

        // Entries. v2 layout adds a single `modality_id: u8` between
        // the per-entry checksum and the stats length prefix. Since we
        // always write the v2 catalog header now, every entry carries
        // its modality_id (zero for single-modality files).
        for entry in &self.entries {
            let name_bytes = entry.name.as_bytes();
            buf.write_u16::<LittleEndian>(name_bytes.len() as u16)?;
            buf.write_all(name_bytes)?;
            buf.write_u64::<LittleEndian>(entry.offset)?;
            buf.write_u64::<LittleEndian>(entry.length)?;
            buf.write_u8(entry.section_type as u8)?;
            buf.write_all(&entry.checksum)?;
            buf.write_u8(entry.modality_id)?;

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

            // v2 catalogs carry a `modality_id: u8` after the
            // per-entry checksum. v1 catalogs don't — leave at 0.
            let modality_id = if catalog_version >= 2 {
                cur.read_u8()?
            } else {
                0u8
            };

            let stats_len = cur.read_u16::<LittleEndian>()? as usize;
            let stats = if stats_len > 0 {
                let mut stats_bytes = vec![0u8; stats_len];
                cur.read_exact(&mut stats_bytes)?;
                let mut stats_cur = std::io::Cursor::new(&stats_bytes);
                Some(ShardStats::read_from(&mut stats_cur, catalog_version)?)
            } else {
                None
            };

            // Skip unknown section types (forward-compatibility)
            let section_type = match SectionType::from_u8(section_type_raw) {
                Some(st) => st,
                None => continue,
            };

            // v1 → v2 axis-overload reconciliation for CSC entries:
            // legacy v1 catalogs encoded `col_start`/`col_end` in the
            // overloaded `row_start`/`row_end` fields. Copy them into
            // the explicit `col_start`/`col_end` slots so consumers can
            // always use `col_range()` regardless of catalog version.
            //
            // CSR/Layer/Obsp entries on v1 catalogs leave
            // `col_start`/`col_end` as 0; the existing accessor surface
            // never reads `col_range()` on row-major shards, so this is
            // safe. Production readers that need the full v2-shape
            // stats can call `reconcile_v1_csr_col_range(n_vars)` after
            // reading.
            let mut entry = FullCatalogEntry {
                name,
                offset,
                length,
                section_type,
                checksum,
                modality_id,
                stats,
            };
            if catalog_version < 2 {
                if let Some(ref mut s) = entry.stats {
                    if matches!(section_type, SectionType::CscShard) {
                        s.col_start = s.row_start;
                        s.col_end = s.row_end;
                    }
                }
            }
            entries.push(entry);
        }

        Ok(Self {
            catalog_version,
            manifest_sequence,
            prev_catalog_offset,
            n_obs,
            entries,
        })
    }

    /// Populate `col_start`/`col_end` for v1 CSR/Layer/Obsp entries
    /// using `n_vars` from the file header. Called by production
    /// readers after `read_from` to fill in the v2-shape stats. CSC
    /// entries are already reconciled inside `read_from` (no `n_vars`
    /// required). No-op on v2 catalogs.
    pub fn reconcile_v1_csr_col_range(&mut self, n_vars: u64) {
        if self.catalog_version >= 2 {
            return;
        }
        for entry in &mut self.entries {
            if !matches!(
                entry.section_type,
                SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
            ) {
                continue;
            }
            if let Some(ref mut s) = entry.stats {
                s.col_start = 0;
                s.col_end = n_vars;
            }
        }
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
    ///
    /// Note: this name is retained for backward compatibility but the
    /// new `csr_shards_sorted()` form is preferred for readability when
    /// CSC sidecars are also present.
    pub fn shards_sorted(&self) -> Vec<&FullCatalogEntry> {
        self.csr_shards_sorted()
    }

    // -----------------------------------------------------------------------
    // Phase C.2: per-modality filtering helpers
    // -----------------------------------------------------------------------

    /// Return all entries belonging to a given modality (any
    /// section_type). `modality_id == 0` returns global entries (the
    /// implicit modality of v1 / single-modality v2 files).
    pub fn shards_for_modality(&self, modality_id: u8) -> Vec<&FullCatalogEntry> {
        self.entries
            .iter()
            .filter(|e| e.modality_id == modality_id)
            .collect()
    }

    /// Return CSR shards belonging to a given modality, sorted by
    /// `row_start`.
    pub fn csr_shards_for_modality(&self, modality_id: u8) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::CsrShard))
        });
        shards
    }

    /// Return CSC shards belonging to a given modality, sorted by
    /// `col_start`.
    pub fn csc_shards_for_modality(&self, modality_id: u8) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard && e.modality_id == modality_id)
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::CscShard))
        });
        shards
    }

    /// Return Layer-CSR shards belonging to `(modality_id, layer_name)`.
    /// Layer section names follow the pattern
    /// `layer/{modality_name}/{layer_name}/shard_{idx}`. Sorted by
    /// `row_start`.
    pub fn layer_csr_shards_for_modality(
        &self,
        modality_id: u8,
        layer_name: &str,
    ) -> Vec<&FullCatalogEntry> {
        let needle = format!("/{layer_name}/");
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard
                    && e.modality_id == modality_id
                    && e.name.contains(&needle)
            })
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::LayerCsrShard))
        });
        shards
    }

    /// Return Layer-CSC shards belonging to `(modality_id, layer_name)`.
    /// Sorted by `col_start`.
    pub fn layer_csc_shards_for_modality(
        &self,
        modality_id: u8,
        layer_name: &str,
    ) -> Vec<&FullCatalogEntry> {
        let needle = format!("/{layer_name}/");
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCscShard
                    && e.modality_id == modality_id
                    && e.name.contains(&needle)
            })
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::LayerCscShard))
        });
        shards
    }

    /// Return CSR shard entries sorted by `stats.row_start`.
    /// Entries without stats are placed at the end.
    pub fn csr_shards_sorted(&self) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::CsrShard))
        });
        shards
    }

    /// Return CSC shard entries sorted by `stats.col_start`. Entries
    /// without stats go at the end.
    pub fn csc_shards_sorted(&self) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard)
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::CscShard))
        });
        shards
    }

    /// Return CSC shard entries whose `[col_start, col_end)` range
    /// intersects the requested half-open interval `[c_lo, c_hi)`.
    /// Sorted by `col_start`. Used for column-range pushdown
    /// ( / E.2).
    pub fn csc_shards_for_col_range(&self, c_lo: u64, c_hi: u64) -> Vec<&FullCatalogEntry> {
        if c_lo >= c_hi {
            return Vec::new();
        }
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                if e.section_type != SectionType::CscShard {
                    return false;
                }
                match &e.stats {
                    Some(s) => {
                        s.major_start(SectionType::CscShard) < c_hi
                            && s.major_end(SectionType::CscShard) > c_lo
                    }
                    None => false,
                }
            })
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::CscShard))
        });
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
            catalog_version: CURRENT_CATALOG_VERSION,
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
        let parsed =
            FullCatalog::read_from(&mut std::io::Cursor::new(&buf), total_len, true).unwrap();
        assert_eq!(parsed.catalog_version, 2);
        assert_eq!(parsed.n_obs, v1.n_obs);
        assert_eq!(parsed.entries.len(), v1.entries.len());
    }
}
