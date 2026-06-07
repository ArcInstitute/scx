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
    /// mismatch (see [`crate::backed::BackedCscReader`]). `0` on v1–v3.
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
    /// entry list. The declared `catalog_version` is upgraded to 4 only
    /// when a non-zero counter is carried; a fully-zero pair (the common
    /// no-CSC / legacy case) keeps the declared version at its v2+ floor
    /// and emits **no** trailing bytes, so existing zero-counter files are
    /// byte-identical to before. v4 catalogs carry the 16 bytes; v<4 do
    /// not — `read_from` keys the trailing read strictly on
    /// `catalog_version >= 4`.
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
        let err =
            LazyShardStats::read_from(&mut Cursor::new(&buf), CURRENT_CATALOG_VERSION, buf.len())
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
        let parsed =
            FullCatalog::read_from(&mut std::io::Cursor::new(&buf), total_len, true).unwrap();
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
        let parsed =
            FullCatalog::read_from(&mut Cursor::new(&buf_zero), buf_zero.len(), true).unwrap();
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
        let err = FullCatalog::read_from(&mut std::io::Cursor::new(&payload), total_len, false)
            .unwrap_err();
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
        let err = FullCatalog::read_from(&mut std::io::Cursor::new(&payload), total_len, false)
            .unwrap_err();
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
        let err = FullCatalog::read_from(&mut std::io::Cursor::new(&payload), total_len, false)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("name truncated") || msg.contains("UnexpectedEof"),
            "expected name-truncated error, got: {msg}"
        );
    }
}
