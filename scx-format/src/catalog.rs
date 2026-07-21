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
    ///
    /// The root catalog is bounded to [`ROOT_CATALOG_MAX_SIZE`] bytes, so a
    /// trustworthy count can be at most `ROOT_CATALOG_MAX_SIZE /
    /// ROOT_CATALOG_ENTRY_SIZE` entries. A corrupt count is rejected up front
    /// (SCX-007) rather than driving a large `with_capacity` and reading past
    /// the root region into section-body bytes.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        const MAX_ROOT_ENTRIES: usize = ROOT_CATALOG_MAX_SIZE / ROOT_CATALOG_ENTRY_SIZE;
        let n_section_groups = r.read_u16::<LittleEndian>()?;
        if n_section_groups as usize > MAX_ROOT_ENTRIES {
            return Err(ScxError::InvalidCatalog(format!(
                "root catalog declares {n_section_groups} entries, exceeds max {MAX_ROOT_ENTRIES} \
                 for the {ROOT_CATALOG_MAX_SIZE}-byte root region"
            )));
        }
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
                // Checked wire-width conversion (SCX-006).
                let bitset_len = u16::try_from(bitset.len()).map_err(|_| {
                    ScxError::InvalidCatalog(format!(
                        "category bitset is {} bytes, exceeds u16::MAX",
                        bitset.len()
                    ))
                })?;
                w.write_u16::<LittleEndian>(bitset_len)?;
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

    /// Build stats carrying only a row range, for metadata shards
    /// (`ObsMetadataShard` / `VarMetadataShard`). These shards have no
    /// CSR value/nnz semantics, so the column range, nnz, value
    /// summaries, and per-column stats are all zero/empty. The row range
    /// lets the query engine map a metadata shard to its global rows
    /// (and skip shards that don't overlap surviving CSR shards) without
    /// decoding the shard payload.
    pub fn row_range_only(row_start: u64, n_rows: u64) -> Self {
        ShardStats {
            row_start,
            row_end: row_start + n_rows,
            col_start: 0,
            col_end: 0,
            nnz: 0,
            value_min: 0,
            value_max: 0,
            value_sum: 0,
            n_indexed_columns: 0,
            column_stats: Vec::new(),
        }
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
        // Checked wire-width conversion (SCX-006): >255 indexed columns would
        // wrap the u8 count and desync the stats blob on read-back.
        let n_stats = u8::try_from(self.column_stats.len())
            .map_err(|_| ScxError::ColumnStatsOverflow(self.column_stats.len()))?;
        w.write_u8(n_stats)?;
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
// LazyShardStats — eager scalars, lazy column_stats decode
// ---------------------------------------------------------------------------

/// `ShardStats` whose per-column statistics payload is retained as
/// opaque bytes and parsed on demand. All fixed-width scalar fields
/// (`row_*`, `col_*`, `nnz`, `value_*`, `n_indexed_columns`) are
/// decoded eagerly at parse time — they are cheap and most read-path
/// consumers need them. The variable-length `column_stats` tail is
/// consumed only by `scx-engine::pushdown` during predicate
/// evaluation; callers without a predicate (the dominant case) never
/// allocate the `Vec<ColumnStat>` or the per-`CategoryBitset`
/// `Vec<u8>` payloads.
///
/// Compared to `ShardStats`:
///
/// - Files with `n_indexed_columns == 0` retain an empty
///   `column_stats_bytes`; `decode_column_stats()` returns `Vec::new`
///   without I/O. The byte-level cost matches the existing
///   `ShardStats::read_from` for these files.
/// - Files with column stats pay the column-stats payload **copy**
///   (into `Box<[u8]>`) eagerly but defer the per-stat
///   `Vec<ColumnStat>` decoding — the larger of the two costs — until
///   `decode_column_stats()` or `to_full()` is called.
///
/// The type does NOT replace `ShardStats` in public APIs; it is a
/// sibling. Callers that need the full eagerly-parsed struct (writer
/// round-trips, validation, metadata inspection) keep using
/// `ShardStats::read_from`. Callers that read the column stats only
/// sometimes (the predicate-pushdown path, future reader integrations)
/// can adopt `LazyShardStats` to skip the per-entry column-stats
/// allocation on the cold case.
#[derive(Debug, Clone, PartialEq)]
pub struct LazyShardStats {
    pub row_start: u64,
    pub row_end: u64,
    pub col_start: u64,
    pub col_end: u64,
    pub nnz: u64,
    pub value_min: u32,
    pub value_max: u32,
    pub value_sum: u64,
    pub n_indexed_columns: u8,
    /// Raw bytes of the `column_stats` payload. Empty when
    /// `n_indexed_columns == 0`. Decoded on demand via
    /// `decode_column_stats()`. Owned (`Box<[u8]>`) so the type can be
    /// freely cloned and shared without lifetime constraints; the
    /// per-stats copy is bounded by the actual `column_stats` size,
    /// not the full stats payload.
    column_stats_bytes: Box<[u8]>,
    /// Catalog version snapshot. Needed at decode time because v1
    /// `ColumnStat` layout is identical to v2 (Phase 1 column stats
    /// did not exist on v1 in practice, but the field-level decoder
    /// branches on it for forward-compat).
    catalog_version: u16,
}

impl LazyShardStats {
    /// Parse from a stats payload reader. `stats_payload_len` is the
    /// `stats_len: u16` value the outer catalog parser already
    /// extracted — it bounds the column-stats tail without requiring
    /// another walk through `ColumnStat::read_from`.
    ///
    /// Returns an error if `stats_payload_len` is smaller than the
    /// fixed-width prefix for the given catalog version.
    pub fn read_from<R: Read>(
        r: &mut R,
        catalog_version: u16,
        stats_payload_len: usize,
    ) -> Result<Self> {
        let prefix_len = if catalog_version >= 2 {
            SHARD_STATS_BASE_SIZE_V2
        } else {
            SHARD_STATS_BASE_SIZE_V1
        };
        if stats_payload_len < prefix_len {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "stats payload too short for v{}: have {}, need >= {}",
                    catalog_version, stats_payload_len, prefix_len
                ),
            )));
        }

        let row_start = r.read_u64::<LittleEndian>()?;
        let row_end = r.read_u64::<LittleEndian>()?;
        let (col_start, col_end) = if catalog_version >= 2 {
            let cs = r.read_u64::<LittleEndian>()?;
            let ce = r.read_u64::<LittleEndian>()?;
            (cs, ce)
        } else {
            (0u64, 0u64)
        };
        let nnz = r.read_u64::<LittleEndian>()?;
        let value_min = r.read_u32::<LittleEndian>()?;
        let value_max = r.read_u32::<LittleEndian>()?;
        let value_sum = r.read_u64::<LittleEndian>()?;
        let n_indexed_columns = r.read_u8()?;

        let tail_len = stats_payload_len - prefix_len;
        let column_stats_bytes = if tail_len == 0 {
            // Common case: `n_indexed_columns == 0`. No heap
            // allocation for the column-stats tail.
            Vec::new().into_boxed_slice()
        } else {
            let mut buf = vec![0u8; tail_len];
            r.read_exact(&mut buf)?;
            buf.into_boxed_slice()
        };

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
            column_stats_bytes,
            catalog_version,
        })
    }

    /// Return the retained column-stats byte payload (empty when
    /// `n_indexed_columns == 0`). Exposed primarily for diagnostics
    /// and tests; production callers should use `decode_column_stats`.
    pub fn column_stats_bytes(&self) -> &[u8] {
        &self.column_stats_bytes
    }

    /// Snapshot of the catalog version that produced this stats
    /// payload. Required for forward-compat decoding of column stats
    /// when v2-only `ColumnStat` variants are added in future
    /// catalog versions.
    pub fn catalog_version(&self) -> u16 {
        self.catalog_version
    }

    /// Decode the retained `column_stats` bytes into a fresh
    /// `Vec<ColumnStat>`. Returns `Vec::new` for files with
    /// `n_indexed_columns == 0` (without touching the byte buffer).
    /// Allocates `n_indexed_columns` × `ColumnStat` + the per-
    /// `CategoryBitset` `Vec<u8>` payloads — i.e., exactly what the
    /// eager `ShardStats::read_from` would have allocated.
    pub fn decode_column_stats(&self) -> Result<Vec<ColumnStat>> {
        if self.n_indexed_columns == 0 {
            return Ok(Vec::new());
        }
        let mut cur: &[u8] = &self.column_stats_bytes;
        let mut out = Vec::with_capacity(self.n_indexed_columns as usize);
        for _ in 0..self.n_indexed_columns {
            out.push(ColumnStat::read_from(&mut cur)?);
        }
        Ok(out)
    }

    /// Force full `ShardStats` decoding for metadata inspection,
    /// validation, or writer round-trips. Allocates the
    /// `column_stats` `Vec`.
    pub fn to_full(&self) -> Result<ShardStats> {
        Ok(ShardStats {
            row_start: self.row_start,
            row_end: self.row_end,
            col_start: self.col_start,
            col_end: self.col_end,
            nnz: self.nnz,
            value_min: self.value_min,
            value_max: self.value_max,
            value_sum: self.value_sum,
            n_indexed_columns: self.n_indexed_columns,
            column_stats: self.decode_column_stats()?,
        })
    }

    /// Build a `LazyShardStats` from an already-decoded `ShardStats`.
    /// Re-serialises the `column_stats` payload into the retained
    /// byte buffer so `to_full()` round-trips byte-identically.
    /// Useful for callers that hold a fully-decoded `ShardStats` and
    /// want to construct a lazy version for downstream APIs.
    pub fn from_full(full: &ShardStats, catalog_version: u16) -> Result<Self> {
        let mut buf = Vec::new();
        for cs in &full.column_stats {
            cs.write_to(&mut buf)?;
        }
        Ok(Self {
            row_start: full.row_start,
            row_end: full.row_end,
            col_start: full.col_start,
            col_end: full.col_end,
            nnz: full.nnz,
            value_min: full.value_min,
            value_max: full.value_max,
            value_sum: full.value_sum,
            n_indexed_columns: full.n_indexed_columns,
            column_stats_bytes: buf.into_boxed_slice(),
            catalog_version,
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
///
/// v3 is a pure forward-compat signal: it doesn't change the catalog
/// wire format (no new fields, no new layout branches). It declares
/// that the writer may have emitted the row-sharded obs/var metadata
/// section types ([`SectionType::ObsMetadataShard`] /
/// [`SectionType::VarMetadataShard`]) introduced for atlas-scale merge
/// and append. Older readers that don't know those types skip them via
/// the existing unknown-section-type warning in
/// [`FullCatalog::read_from`] and then fail loudly when callers ask for
/// the global obs section that no longer exists — preferable to a
/// silent partial read. New readers branching on `catalog_version >= 3`
/// can short-circuit to the sharded paths without re-scanning the
/// catalog.
///
/// v4 appends two `u64` generation counters (`data_generation`,
/// `csc_build_generation`) immediately after the entry list, before the
/// trailing checksum. They are a CSC-sidecar freshness guard: see the
/// field docs on [`FullCatalog`]. The append is purely additive — older
/// readers parse the entry list, ignore the trailing 16 bytes, and the
/// checksum still validates over the whole payload (the extra bytes are
/// inside the checksummed region). New readers read the counters when
/// `catalog_version >= 4`; v1–v3 catalogs lack the fields and default both
/// to `0` (a truncated v4 catalog missing the bytes is corrupt and surfaces
/// as a read error). A `0 == 0` match means "fresh", so v1–v3 files (which
/// lack the counters) are never treated as stale.
pub const CURRENT_CATALOG_VERSION: u16 = 4;

#[derive(Debug, Clone)]
pub struct FullCatalog {
    pub catalog_version: u16,
    pub manifest_sequence: u64,
    pub prev_catalog_offset: u64,
    pub n_obs: u64,
    pub entries: Vec<FullCatalogEntry>,
    /// Monotonic identity of the CSR X data (v4+). Bumped by every
    /// in-place CSR-content-mutating writer (`append`/`compact`/`merge`);
    /// **not** bumped by CSC-only rewrites (`build-csc`/`--rebuild-csc`).
    /// `subset` writes a brand-new file at the default generation (1)
    /// rather than bumping a source. `0` on v1–v3 catalogs (the field was
    /// absent).
    pub data_generation: u64,
    /// The `data_generation` value the current CSC sidecar was built
    /// against (v4+), or `0` when there is no sidecar. A sidecar is fresh
    /// iff `csc_build_generation == data_generation`; readers reject a
    /// mismatch (see `scx_format_io::BackedCscReader`). `0` on v1–v3.
    pub csc_build_generation: u64,
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
    ///
    /// The two v4 generation counters ride in 16 trailing bytes after the
    /// entry list, emitted whenever the declared `catalog_version >= 4`.
    /// A non-zero counter forces the declared version up to 4; if the
    /// caller declares a lower version AND both counters are zero, no
    /// trailing bytes are written (the legacy/v2-floor path, kept so a
    /// hand-constructed v2 catalog still round-trips byte-identically).
    /// Note the shipped writer always passes `CURRENT_CATALOG_VERSION`
    /// (= 4, `writer.rs`), so production catalogs always carry the 16
    /// trailing bytes even with zero counters. `read_from` keys the
    /// trailing read strictly on `catalog_version >= 4`, so read-back is
    /// self-consistent in every case.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut catalog_version = std::cmp::max(self.catalog_version, 2);
        if self.data_generation != 0 || self.csc_build_generation != 0 {
            catalog_version = std::cmp::max(catalog_version, 4);
        }

        let mut buf = Vec::new();

        // Header fields
        buf.write_u16::<LittleEndian>(catalog_version)?;
        buf.write_u64::<LittleEndian>(self.manifest_sequence)?;
        buf.write_u64::<LittleEndian>(self.prev_catalog_offset)?;
        buf.write_u64::<LittleEndian>(self.n_obs)?;
        // Checked wire-width conversions (SCX-006): a value that overflows the
        // serialized field must fail loudly at write time, never silently
        // narrow and desynchronize the reader.
        let n_entries = u32::try_from(self.entries.len()).map_err(|_| {
            ScxError::InvalidCatalog(format!(
                "catalog has {} entries, exceeds u32::MAX",
                self.entries.len()
            ))
        })?;
        buf.write_u32::<LittleEndian>(n_entries)?;

        // Entries. v2 layout adds a single `modality_id: u8` between
        // the per-entry checksum and the stats length prefix. Since we
        // always write the v2 catalog header now, every entry carries
        // its modality_id (zero for single-modality files).
        for entry in &self.entries {
            let name_bytes = entry.name.as_bytes();
            let name_len = u16::try_from(name_bytes.len()).map_err(|_| {
                ScxError::InvalidCatalog(format!(
                    "section name '{}' is {} bytes, exceeds u16::MAX",
                    entry.name,
                    name_bytes.len()
                ))
            })?;
            buf.write_u16::<LittleEndian>(name_len)?;
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
                    let stats_len = u16::try_from(stats_buf.len()).map_err(|_| {
                        ScxError::InvalidCatalog(format!(
                            "stats blob for section '{}' is {} bytes, exceeds u16::MAX",
                            entry.name,
                            stats_buf.len()
                        ))
                    })?;
                    buf.write_u16::<LittleEndian>(stats_len)?;
                    buf.write_all(&stats_buf)?;
                }
                None => {
                    buf.write_u16::<LittleEndian>(0)?;
                }
            }
        }

        // v4 trailing generation counters (only when declared v4 — i.e.
        // a counter is non-zero). Inside the checksummed payload.
        if catalog_version >= 4 {
            buf.write_u64::<LittleEndian>(self.data_generation)?;
            buf.write_u64::<LittleEndian>(self.csc_build_generation)?;
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
            return Err(ScxError::ChecksumMismatch {
                section: "full_catalog (too short)".to_string(),
            });
        }

        let mut all_bytes = vec![0u8; total_len];
        r.read_exact(&mut all_bytes)?;

        let payload_len = total_len - 32;
        let (payload, expected_checksum) = all_bytes.split_at(payload_len);

        if verify_checksum {
            let computed = blake3_hash(payload);
            if computed[..] != *expected_checksum {
                return Err(ScxError::ChecksumMismatch {
                    section: "full_catalog".to_string(),
                });
            }
        }

        // Parse the payload through a `&mut &[u8]` reader so we can
        // peel off name and stats payloads as borrowed sub-slices of
        // `payload` without copying into per-entry `Vec<u8>` buffers
        // or wrapping them in nested `Cursor`s.
        let mut cur: &[u8] = payload;
        let catalog_version = cur.read_u16::<LittleEndian>()?;
        let manifest_sequence = cur.read_u64::<LittleEndian>()?;
        let prev_catalog_offset = cur.read_u64::<LittleEndian>()?;
        let n_obs = cur.read_u64::<LittleEndian>()?;
        let n_entries = cur.read_u32::<LittleEndian>()? as usize;

        // Minimum serialized size of a single v2 catalog entry:
        // 2 (name_len) + 0 (empty name) + 8 (offset) + 8 (length) +
        // 1 (section_type) + 32 (checksum) + 1 (modality_id) +
        // 2 (stats_len) = 54 bytes.
        // v1 entries lack modality_id: 53 bytes. Use the smaller bound.
        const MIN_ENTRY_BYTES: usize = 53;
        crate::error::validate_allocation(n_entries.saturating_mul(MIN_ENTRY_BYTES), payload_len)?;

        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            let name_len = cur.read_u16::<LittleEndian>()? as usize;
            // Borrow the name bytes directly out of the outer payload
            // slice — no per-entry `vec![0u8; name_len]` copy. UTF-8
            // is validated in place against the borrowed slice; the
            // single `String` allocation below replaces the previous
            // (zero-init Vec + into-String) pair.
            let (name_bytes, rest) = cur.split_at_checked(name_len).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "catalog entry name truncated",
                )
            })?;
            cur = rest;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                .to_string();

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
                // Parse `ShardStats` directly out of a borrowed
                // sub-slice — no `vec![0u8; stats_len]` copy, no
                // nested `Cursor`. The outer reader advances by
                // exactly `stats_len` bytes regardless of how many
                // `ShardStats::read_from` consumes, preserving the
                // forward-compat property of the length prefix.
                let (stats_bytes, rest) = cur.split_at_checked(stats_len).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "catalog entry stats payload truncated",
                    )
                })?;
                cur = rest;
                let mut stats_reader: &[u8] = stats_bytes;
                Some(ShardStats::read_from(&mut stats_reader, catalog_version)?)
            } else {
                None
            };

            // Skip unknown section types (forward-compatibility) with a
            // warning, matching the spec requirement that readers skip
            // unrecognised types rather than failing.
            let section_type = match SectionType::from_u8(section_type_raw) {
                Some(st) => st,
                None => {
                    log::warn!(
                        "skipping unknown section type {} (name: '{}') at offset {}",
                        section_type_raw,
                        name,
                        offset,
                    );
                    continue;
                }
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

        // v4 trailing generation counters. Present only when the catalog
        // declares v4; v1–v3 catalogs default both to 0 (a `0 == 0`
        // freshness match, so legacy CSC sidecars are never rejected).
        let (data_generation, csc_build_generation) = if catalog_version >= 4 {
            let data_generation = cur.read_u64::<LittleEndian>()?;
            let csc_build_generation = cur.read_u64::<LittleEndian>()?;
            (data_generation, csc_build_generation)
        } else {
            (0, 0)
        };

        Ok(Self {
            catalog_version,
            manifest_sequence,
            prev_catalog_offset,
            n_obs,
            entries,
            data_generation,
            csc_build_generation,
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

    /// De-duplicated logical names under `<prefix>/`, considering both the
    /// legacy single-section type and the sharded type (stripping the
    /// `_shard_N` suffix). Used by obsm/varm/obsp/varp enumeration on every
    /// reader. Pure catalog scan, no I/O.
    pub fn list_logical_names(
        &self,
        prefix: &str,
        single_type: SectionType,
        shard_type: SectionType,
    ) -> Vec<String> {
        let path_prefix = format!("{prefix}/");
        let shard_marker = "_shard_";
        let mut names: std::collections::BTreeSet<String> = Default::default();
        for entry in &self.entries {
            if entry.section_type == single_type {
                if let Some(name) = entry.name.strip_prefix(&path_prefix) {
                    names.insert(name.to_string());
                }
            } else if entry.section_type == shard_type {
                if let Some(rest) = entry.name.strip_prefix(&path_prefix) {
                    if let Some(idx) = rest.rfind(shard_marker) {
                        names.insert(rest[..idx].to_string());
                    }
                }
            }
        }
        names.into_iter().collect()
    }

    /// De-duplicated layer names from `LayerCsrShard` entries (strips the
    /// `_shard_N` suffix). Pure catalog scan, no I/O.
    pub fn layer_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::LayerCsrShard)
            .filter_map(|e| e.name.rfind("_shard_").map(|pos| e.name[..pos].to_string()))
            .collect();
        names.sort();
        names.dedup();
        names
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

    /// Whether the CSR shards of `modality_id` form disjoint
    /// `[row_start, row_end)` ranges that exactly tile `[0, n_obs)`.
    ///
    /// This is the invariant every multimodal query path relies on: each
    /// modality independently covers the shared global obs axis (empty CSR
    /// rows encode "no measurement"), so a global obs row mask can be
    /// applied to any single modality's shard list without cross-modality
    /// interleaving. Returns `false` if any shard lacks row-range stats, if
    /// two ranges overlap, or if the union leaves a gap or overshoots
    /// `n_obs`. An empty shard list is "covering" only when `n_obs == 0`.
    ///
    /// Intended for `debug_assert!` at modality-scoped scan sites and for
    /// property tests; it is O(n_shards · log n_shards) and does no I/O.
    pub fn modality_csr_ranges_tile_obs(&self, modality_id: u8, n_obs: u64) -> bool {
        let shards = self.csr_shards_for_modality(modality_id);
        let mut expected: u64 = 0;
        for entry in &shards {
            let Some(stats) = entry.stats.as_ref() else {
                return false;
            };
            let start = stats.major_start(SectionType::CsrShard);
            let end = stats.major_end(SectionType::CsrShard);
            // Sorted by row_start; a contiguous tiling requires each shard
            // to begin exactly where the previous one ended.
            if start != expected || end < start {
                return false;
            }
            expected = end;
        }
        expected == n_obs
    }

    /// Phase 5b: return detection-bitmap shards belonging to a given
    /// modality, sorted by `row_start`. Mirrors
    /// [`Self::csr_shards_for_modality`] but filters on
    /// `SectionType::BitmapShard`.
    pub fn bitmap_shards_for_modality(&self, modality_id: u8) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::BitmapShard && e.modality_id == modality_id)
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::BitmapShard))
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

    /// Return CSR shard entries sorted by `stats.row_start`, across **all
    /// modalities** (no `modality_id` filter). Entries without stats are placed
    /// at the end.
    ///
    /// **Multimodal caveat:** on a multimodal file every modality independently
    /// tiles the shared obs axis `[0, n_obs)`, so this flattened list contains
    /// **overlapping** `[row_start, row_end)` ranges (e.g. RNA `[0,1000)` and
    /// ADT `[0,1000)` both start at row 0). A shard's **positional index** in
    /// this list therefore does **not** identify an obs row or a modality — do
    /// not key anything obs-indexed (deletion vectors, row masks) by it. Use
    /// [`Self::csr_shards_for_modality`] for per-modality iteration, and
    /// [`Self::has_overlapping_csr_ranges`] to `debug_assert!` a clean single
    /// tiling where one is required.
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

    /// Whether the flattened [`Self::csr_shards_sorted`] list contains
    /// overlapping `[row_start, row_end)` ranges. This is `true` for **any**
    /// multimodal file (each modality independently tiles `[0, n_obs)`), and
    /// `false` for a single clean tiling.
    ///
    /// It is the cross-cutting tripwire against the class of bug this guards:
    /// code that positionally indexes `csr_shards_sorted()` as if it were one
    /// non-overlapping tiling. `debug_assert!(!catalog.has_overlapping_csr_ranges())`
    /// at any such site. O(n_shards) over the already-sorted list; no I/O.
    pub fn has_overlapping_csr_ranges(&self) -> bool {
        let mut max_end: Option<u64> = None;
        for entry in self.csr_shards_sorted() {
            let Some(stats) = entry.stats.as_ref() else {
                continue;
            };
            let start = stats.major_start(SectionType::CsrShard);
            let end = stats.major_end(SectionType::CsrShard);
            if let Some(prev_end) = max_end {
                if start < prev_end {
                    return true;
                }
            }
            max_end = Some(max_end.map_or(end, |m| m.max(end)));
        }
        false
    }

    /// Return `adata.raw` CSR shard entries ([`SectionType::RawCsrShard`])
    /// sorted by `stats.row_start`. Entries without stats are placed at
    /// the end.
    pub fn raw_csr_shards_sorted(&self) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::RawCsrShard)
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(SectionType::RawCsrShard))
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

    /// Return dense-mapping (obsm/varm) shard entries for a given
    /// `(section_type, modality_id, name_prefix)`, sorted by `row_start`.
    /// Entries without stats are placed at the end.
    ///
    /// Unlike the CSR/CSC/layer helpers, the caller supplies the already
    /// formatted `name_prefix` (e.g. `obsm/X_pca_shard_`) because dense
    /// mappings are keyed by a runtime axis prefix plus the user's key.
    pub fn dense_mapping_shards_sorted(
        &self,
        section_type: SectionType,
        modality_id: u8,
        name_prefix: &str,
    ) -> Vec<&FullCatalogEntry> {
        let mut shards: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                e.section_type == section_type
                    && e.modality_id == modality_id
                    && e.name.starts_with(name_prefix)
            })
            .collect();
        shards.sort_by_key(|e| {
            e.stats
                .as_ref()
                .map_or(u64::MAX, |s| s.major_start(section_type))
        });
        shards
    }
}

#[cfg(test)]
#[path = "catalog_tests.rs"]
mod tests;
