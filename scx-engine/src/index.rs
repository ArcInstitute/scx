// Predicate index read/write — docs/format.md (Predicate Indexes)
//
// Implements sorted value-to-shard-range mappings that enable fine-grained
// row-level filtering within shards. This is level 2 of the two-level pushdown
// strategy (docs/api.md (Query engine, optimizations)).

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Write};

use arrow::array::{
    Array, ArrayRef, AsArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{EngineError, Result};

// ============================================================================
// C1. Data structures
// ============================================================================

/// A predicate index section per docs/format.md (Predicate Indexes).
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateIndex {
    pub version: u8, // 1
    pub columns: Vec<IndexedColumn>,
}

/// An indexed column — either categorical or numeric.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexedColumn {
    Categorical(CategoricalIndex),
    Numeric(NumericIndex),
}

/// Categorical index: sorted value-to-shard-range mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalIndex {
    pub column_name: String,
    pub entries: Vec<CategoricalEntry>, // sorted lexicographically by value
}

/// One unique categorical value and the shard ranges containing it.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalEntry {
    pub value: String,
    pub shard_ranges: Vec<ShardRange>,
}

/// A range of rows within a shard.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardRange {
    pub shard_id: u32,
    pub row_start: u32, // within shard, local index
    pub row_end: u32,   // exclusive
}

/// Numeric index: a B+ tree over per-shard value bounds.
///
/// Writers emit at most one leaf entry per shard — none for a shard holding
/// no value (see [`numeric_leaves_from_spans`]).
/// Nothing navigates the tree to answer a range query — `eval_rowset` is
/// residual for every numeric operator — so in practice this is the carrier
/// for the per-shard `[min, max]` that Level-1 pruning reads, and the shape is
/// a B+ tree because the wire format is. Readers accept the finer leaves
/// older files carry.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericIndex {
    pub column_name: String,
    pub fanout: u16,
    pub internal_pages: Vec<InternalPage>,
    pub leaf_pages: Vec<LeafPage>,
}

/// Internal (non-leaf) page of the B+ tree.
#[derive(Debug, Clone, PartialEq)]
pub struct InternalPage {
    pub n_keys: u16,
    pub keys: Vec<f64>,     // split values
    pub children: Vec<u32>, // page indices, length = n_keys + 1
}

/// Leaf page of the B+ tree.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafPage {
    pub entries: Vec<NumericLeafEntry>,
}

/// A conservative bound on a shard row range: every value in
/// `[row_start, row_end)` of `shard_id` lies within `[min_value, max_value]`.
/// It does not record which row holds which value.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericLeafEntry {
    pub min_value: f64,
    pub max_value: f64,
    pub shard_id: u32,
    pub row_start: u32,
    pub row_end: u32,
}

// ============================================================================
// C2. Serialization — docs/format.md (Predicate Indexes) binary layout
// ============================================================================

/// Upper bound on per-column categorical entries the reader will accept
/// from an on-disk u32 length prefix. Above this the input is treated
/// as malformed and rejected as `InvalidData`. A real-world categorical
/// column (cell_type, donor_id, …) has <100k unique values; 10M is
/// already pathological.
const MAX_CATEGORICAL_ENTRIES: usize = 10_000_000;

/// Upper bound on B+ tree internal/leaf pages per numeric index.
const MAX_NUMERIC_PAGES: usize = 10_000_000;

/// Upper bound on leaf entries per single numeric leaf page.
const MAX_NUMERIC_LEAF_ENTRIES_PER_PAGE: usize = 1_000_000;

/// Cap the `Vec::with_capacity` allocation hint for an untrusted count,
/// so that a single crafted u32 cannot trigger a multi-GB up-front
/// allocation. The vec will grow organically beyond the hint as
/// elements are pushed.
fn capacity_hint(requested: usize) -> usize {
    requested.min(1024)
}

/// Reject a count read from an on-disk length prefix if it exceeds a
/// declared upper bound. Returns `InvalidData` with a `descriptor`
/// naming the field for diagnosability.
fn check_count_bound(count: usize, max: usize, descriptor: &str) -> Result<()> {
    if count > max {
        return Err(EngineError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("PredicateIndex {descriptor} count {count} exceeds maximum {max}"),
        )));
    }
    Ok(())
}

/// Defensive writer-side cast: reject a value that doesn't fit in the
/// target on-disk integer width with a descriptive error rather than
/// silently wrapping via `as uN`. Used in the v1 `PredicateIndex` writer
/// where the on-disk format uses `u16` / `u32` for several counters that
/// could in principle be hit at multi-billion-row / extreme-cardinality
/// scale. The v2 encoding (negotiated via the `version` byte) widens
/// these to `u32` / `u64`; for now, surface the limit cleanly.
fn write_cast<T, U>(value: T, descriptor: &str) -> Result<U>
where
    U: TryFrom<T>,
    T: std::fmt::Display + Copy,
{
    U::try_from(value).map_err(|_| {
        EngineError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "PredicateIndex v1 writer: {descriptor} value {value} exceeds the on-disk \
                 narrow-counter width. Use PredicateIndex v2 (wider counters) for inputs \
                 at this scale."
            ),
        ))
    })
}

/// Returns true if any counter in `index` exceeds the v1 (narrow) on-disk
/// widths and the index must therefore be written with the v2 layout
/// (`u32` value lengths and per-value range counts, `u64` entry totals).
/// Pre-scan is O(total entries) but allocates nothing, and lets the
/// writer pick the narrowest correct encoding rather than blindly
/// promoting every file to v2.
fn requires_v2_encoding(index: &PredicateIndex) -> bool {
    if u16::try_from(index.columns.len()).is_err() {
        return true;
    }
    for col in &index.columns {
        match col {
            IndexedColumn::Categorical(cat) => {
                if u16::try_from(cat.column_name.len()).is_err() {
                    return true;
                }
                if u32::try_from(cat.entries.len()).is_err() {
                    return true;
                }
                for entry in &cat.entries {
                    if u16::try_from(entry.value.len()).is_err() {
                        return true;
                    }
                    if u16::try_from(entry.shard_ranges.len()).is_err() {
                        return true;
                    }
                }
            }
            IndexedColumn::Numeric(num) => {
                if u16::try_from(num.column_name.len()).is_err() {
                    return true;
                }
                if u32::try_from(num.leaf_pages.len()).is_err() {
                    return true;
                }
                if u32::try_from(num.internal_pages.len()).is_err() {
                    return true;
                }
                let mut total: u64 = 0;
                for lp in &num.leaf_pages {
                    if u32::try_from(lp.entries.len()).is_err() {
                        return true;
                    }
                    total = total.saturating_add(lp.entries.len() as u64);
                }
                if u32::try_from(total).is_err() {
                    return true;
                }
            }
        }
    }
    false
}

/// Which kind of index, if any, covers a column. Returned by
/// [`PredicateIndex::indexed_kind`] so the query partitioner can decide whether
/// a predicate on a column is resolvable from the index (row-set fast path) or
/// must fall back to decoding obs shards (residual path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    Categorical,
    Numeric,
}

/// Whether `index` covers every obs row, i.e. is safe to use for row-set
/// pushdown on this file.
///
/// The file-scope predicate index can go **stale** after `append`, which adds
/// new obs shards at the tail without rewriting the index (CLAUDE.md, cloud
/// query notes). A stale index would silently miss the appended rows on the
/// row-set fast path. This guard detects that case: the index covers all of obs
/// only when the maximum global row it references reaches `n_obs`.
///
/// Conservative by design: if the trailing obs rows happen to be null in *every*
/// indexed column they aren't referenced by any entry, so a fresh index can also
/// report `< n_obs` and we fall back to the full obs scan. That fallback is
/// correct (just slower) — atlas builds populate the indexed columns, so the
/// fast path applies in practice.
pub fn index_covers_all_obs(
    index: &PredicateIndex,
    obs_shard_ranges: &[(u32, u64, u64)],
    n_obs: u64,
) -> bool {
    if index.columns.is_empty() {
        return false;
    }
    index.max_covered_global_row(obs_shard_ranges) >= n_obs
}

impl PredicateIndex {
    // ------------------------------------------------------------------------
    // Query-time lookups (row-set pushdown). These consume the index for row
    // *selection*, not just the catalog-level shard *pruning* that
    // `build_category_dicts` / `prune_shards_by_catalog_with_dict` do today.
    // ------------------------------------------------------------------------

    /// The shard ranges containing `value` in the indexed categorical column
    /// `column`, or `None` if `column` is not an indexed categorical.
    ///
    /// - `None` => column not indexed as a categorical → the caller treats the
    ///   predicate as residual (decode + mask).
    /// - `Some(&[])` => column is indexed but `value` is absent → the predicate
    ///   matches no rows (an exact, empty row-set). This is *not* residual.
    ///
    /// `CategoricalIndex::entries` is sorted lexicographically by value
    /// (`build_categorical_index` builds it from a `BTreeMap`), so this is a
    /// binary search.
    pub fn categorical_eq(&self, column: &str, value: &str) -> Option<&[ShardRange]> {
        for col in &self.columns {
            if let IndexedColumn::Categorical(cat) = col {
                if cat.column_name == column {
                    return Some(
                        match cat
                            .entries
                            .binary_search_by(|e| e.value.as_str().cmp(value))
                        {
                            Ok(pos) => &cat.entries[pos].shard_ranges,
                            Err(_) => &[],
                        },
                    );
                }
            }
        }
        None
    }

    /// Which kind of index (if any) covers `column`.
    pub fn indexed_kind(&self, column: &str) -> Option<IndexKind> {
        for col in &self.columns {
            match col {
                IndexedColumn::Categorical(cat) if cat.column_name == column => {
                    return Some(IndexKind::Categorical)
                }
                IndexedColumn::Numeric(num) if num.column_name == column => {
                    return Some(IndexKind::Numeric)
                }
                _ => {}
            }
        }
        None
    }

    /// The maximum global row covered by any shard range in this index, mapped
    /// through `obs_shard_ranges` (`(shard_idx, row_start, row_end)` sorted by
    /// `shard_idx`). Used by [`index_covers_all_obs`].
    fn max_covered_global_row(&self, obs_shard_ranges: &[(u32, u64, u64)]) -> u64 {
        let mut max_end = 0u64;
        let mut consider = |shard_id: u32, row_end: u32| {
            if let Ok(pos) = obs_shard_ranges.binary_search_by_key(&shard_id, |(idx, _, _)| *idx) {
                let (_, shard_row_start, shard_row_end) = obs_shard_ranges[pos];
                let g = (shard_row_start + row_end as u64).min(shard_row_end);
                if g > max_end {
                    max_end = g;
                }
            }
        };
        for col in &self.columns {
            match col {
                IndexedColumn::Categorical(cat) => {
                    for entry in &cat.entries {
                        for sr in &entry.shard_ranges {
                            consider(sr.shard_id, sr.row_end);
                        }
                    }
                }
                IndexedColumn::Numeric(num) => {
                    for leaf in &num.leaf_pages {
                        for e in &leaf.entries {
                            consider(e.shard_id, e.row_end);
                        }
                    }
                }
            }
        }
        max_end
    }

    /// Serialize the predicate index per docs/format.md (Predicate Indexes).
    ///
    /// The on-disk version byte is auto-selected: v1 (narrow counters) when
    /// every counter fits the v1 widths (`u16` value lengths and per-value
    /// range counts, `u32` entry totals), v2 otherwise. v2 widens to `u32` /
    /// `u64` so extreme-cardinality columns and >1B-row obs axes can be
    /// represented without overflow. The `version` field on `self` is
    /// ignored — the encoding is determined by the data alone, and the
    /// on-disk byte is the source of truth on read.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        if requires_v2_encoding(self) {
            self.write_to_v2(w)
        } else {
            self.write_to_v1(w)
        }
    }

    /// Write the v1 (narrow-counter) layout. Returns `InvalidData` if a
    /// counter doesn't fit — callers should use [`Self::write_to`] which
    /// auto-routes to v2 in that case.
    fn write_to_v1<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u8(1)?;
        w.write_u16::<LittleEndian>(write_cast::<_, u16>(self.columns.len(), "columns.len()")?)?;
        for col in &self.columns {
            match col {
                IndexedColumn::Categorical(cat) => {
                    // column name
                    let name_bytes = cat.column_name.as_bytes();
                    w.write_u16::<LittleEndian>(write_cast::<_, u16>(
                        name_bytes.len(),
                        "categorical column_name.len()",
                    )?)?;
                    w.write_all(name_bytes)?;
                    // column_type = 0 (categorical)
                    w.write_u8(0)?;
                    // n_entries
                    w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                        cat.entries.len(),
                        "categorical entries.len()",
                    )?)?;
                    for entry in &cat.entries {
                        let val_bytes = entry.value.as_bytes();
                        w.write_u16::<LittleEndian>(write_cast::<_, u16>(
                            val_bytes.len(),
                            "categorical value.len()",
                        )?)?;
                        w.write_all(val_bytes)?;
                        w.write_u16::<LittleEndian>(write_cast::<_, u16>(
                            entry.shard_ranges.len(),
                            "categorical shard_ranges.len()",
                        )?)?;
                        for sr in &entry.shard_ranges {
                            w.write_u32::<LittleEndian>(sr.shard_id)?;
                            w.write_u32::<LittleEndian>(sr.row_start)?;
                            w.write_u32::<LittleEndian>(sr.row_end)?;
                        }
                    }
                }
                IndexedColumn::Numeric(num) => {
                    // column name
                    let name_bytes = num.column_name.as_bytes();
                    w.write_u16::<LittleEndian>(write_cast::<_, u16>(
                        name_bytes.len(),
                        "numeric column_name.len()",
                    )?)?;
                    w.write_all(name_bytes)?;
                    // column_type = 1 (numeric)
                    w.write_u8(1)?;
                    // n_entries (total leaf entries for the header) — checked sum
                    let mut total_entries: u32 = 0;
                    for lp in &num.leaf_pages {
                        let n =
                            write_cast::<_, u32>(lp.entries.len(), "numeric leaf entries.len()")?;
                        total_entries = total_entries.checked_add(n).ok_or_else(|| {
                            EngineError::IoError(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "PredicateIndex v1 writer: numeric total_entries overflows u32; \
                                 use PredicateIndex v2 (u64 entry totals) for inputs at this scale.",
                            ))
                        })?;
                    }
                    w.write_u32::<LittleEndian>(total_entries)?;
                    // B+ tree metadata
                    w.write_u16::<LittleEndian>(num.fanout)?;
                    w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                        num.leaf_pages.len(),
                        "numeric leaf_pages.len()",
                    )?)?;
                    w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                        num.internal_pages.len(),
                        "numeric internal_pages.len()",
                    )?)?;
                    // Internal pages
                    for page in &num.internal_pages {
                        w.write_u16::<LittleEndian>(page.n_keys)?;
                        for &key in &page.keys {
                            w.write_f64::<LittleEndian>(key)?;
                        }
                        for &child in &page.children {
                            w.write_u32::<LittleEndian>(child)?;
                        }
                    }
                    // Leaf pages
                    for page in &num.leaf_pages {
                        w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                            page.entries.len(),
                            "numeric per-page entries.len()",
                        )?)?;
                        for entry in &page.entries {
                            w.write_f64::<LittleEndian>(entry.min_value)?;
                            w.write_f64::<LittleEndian>(entry.max_value)?;
                            w.write_u32::<LittleEndian>(entry.shard_id)?;
                            w.write_u32::<LittleEndian>(entry.row_start)?;
                            w.write_u32::<LittleEndian>(entry.row_end)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Write the v2 (wide-counter) layout. v2 widens the on-disk counters
    /// that v1 stored as `u16` / `u32` to `u32` / `u64`, so extreme-cardinality
    /// columns or multi-billion-row obs axes can be represented without
    /// silent truncation. Wire format:
    ///
    /// - `version = 2` (u8)
    /// - `n_columns` (u32)            ← was u16 in v1
    /// - per column:
    ///   - `name_len` (u32)           ← was u16
    ///   - name bytes
    ///   - `column_type` (u8)
    ///   - `n_entries` (u64)          ← was u32
    ///   - categorical:
    ///     - `val_len` (u32)          ← was u16
    ///     - value bytes
    ///     - `n_ranges` (u32)         ← was u16
    ///     - per range: shard_id/row_start/row_end (u32 each, unchanged)
    ///   - numeric:
    ///     - `fanout` (u16, unchanged)
    ///     - `n_leaf_pages` (u64)     ← was u32
    ///     - `n_internal_pages` (u64) ← was u32
    ///     - internal pages: n_keys (u16), keys (f64), children (u32) [unchanged]
    ///     - per leaf page: `n_entries` (u64) ← was u32, then entries [unchanged]
    fn write_to_v2<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u8(2)?;
        w.write_u32::<LittleEndian>(write_cast::<_, u32>(self.columns.len(), "columns.len()")?)?;
        for col in &self.columns {
            match col {
                IndexedColumn::Categorical(cat) => {
                    let name_bytes = cat.column_name.as_bytes();
                    w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                        name_bytes.len(),
                        "categorical column_name.len()",
                    )?)?;
                    w.write_all(name_bytes)?;
                    w.write_u8(0)?;
                    w.write_u64::<LittleEndian>(cat.entries.len() as u64)?;
                    for entry in &cat.entries {
                        let val_bytes = entry.value.as_bytes();
                        w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                            val_bytes.len(),
                            "categorical value.len()",
                        )?)?;
                        w.write_all(val_bytes)?;
                        w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                            entry.shard_ranges.len(),
                            "categorical shard_ranges.len()",
                        )?)?;
                        for sr in &entry.shard_ranges {
                            w.write_u32::<LittleEndian>(sr.shard_id)?;
                            w.write_u32::<LittleEndian>(sr.row_start)?;
                            w.write_u32::<LittleEndian>(sr.row_end)?;
                        }
                    }
                }
                IndexedColumn::Numeric(num) => {
                    let name_bytes = num.column_name.as_bytes();
                    w.write_u32::<LittleEndian>(write_cast::<_, u32>(
                        name_bytes.len(),
                        "numeric column_name.len()",
                    )?)?;
                    w.write_all(name_bytes)?;
                    w.write_u8(1)?;
                    let total_entries: u64 = num
                        .leaf_pages
                        .iter()
                        .map(|lp| lp.entries.len() as u64)
                        .sum();
                    w.write_u64::<LittleEndian>(total_entries)?;
                    w.write_u16::<LittleEndian>(num.fanout)?;
                    w.write_u64::<LittleEndian>(num.leaf_pages.len() as u64)?;
                    w.write_u64::<LittleEndian>(num.internal_pages.len() as u64)?;
                    for page in &num.internal_pages {
                        w.write_u16::<LittleEndian>(page.n_keys)?;
                        for &key in &page.keys {
                            w.write_f64::<LittleEndian>(key)?;
                        }
                        for &child in &page.children {
                            w.write_u32::<LittleEndian>(child)?;
                        }
                    }
                    for page in &num.leaf_pages {
                        w.write_u64::<LittleEndian>(page.entries.len() as u64)?;
                        for entry in &page.entries {
                            w.write_f64::<LittleEndian>(entry.min_value)?;
                            w.write_f64::<LittleEndian>(entry.max_value)?;
                            w.write_u32::<LittleEndian>(entry.shard_id)?;
                            w.write_u32::<LittleEndian>(entry.row_start)?;
                            w.write_u32::<LittleEndian>(entry.row_end)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Deserialize a predicate index from a reader.
    ///
    /// Defends against malformed input: the u32 length-prefix fields are
    /// each capped, both via a hard upper bound (`InvalidData` error if
    /// exceeded) and via a capped `Vec::with_capacity` hint so a single
    /// crafted u32 cannot trigger a multi-GB allocation before the inner
    /// `read_exact` calls EOF. Plausible legitimate inputs stay well
    /// below the caps.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let version = r.read_u8()?;
        match version {
            1 => Self::read_from_v1(r),
            2 => Self::read_from_v2(r),
            v => Err(EngineError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown predicate index version: {v}"),
            ))),
        }
    }

    fn read_from_v1<R: Read>(r: &mut R) -> Result<Self> {
        let version = 1u8;
        let n_columns = r.read_u16::<LittleEndian>()?;
        let mut columns = Vec::with_capacity(n_columns as usize);

        for _ in 0..n_columns {
            // column name
            let name_len = r.read_u16::<LittleEndian>()? as usize;
            let mut name_bytes = vec![0u8; name_len];
            r.read_exact(&mut name_bytes)?;
            let column_name = String::from_utf8(name_bytes).map_err(|e| {
                EngineError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })?;

            let column_type = r.read_u8()?;
            let _n_entries = r.read_u32::<LittleEndian>()?;

            match column_type {
                0 => {
                    // Categorical
                    let n_cat_entries = _n_entries as usize;
                    check_count_bound(
                        n_cat_entries,
                        MAX_CATEGORICAL_ENTRIES,
                        "categorical entries",
                    )?;
                    let mut entries = Vec::with_capacity(capacity_hint(n_cat_entries));
                    for _ in 0..n_cat_entries {
                        let val_len = r.read_u16::<LittleEndian>()? as usize;
                        let mut val_bytes = vec![0u8; val_len];
                        r.read_exact(&mut val_bytes)?;
                        let value = String::from_utf8(val_bytes).map_err(|e| {
                            EngineError::IoError(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                e,
                            ))
                        })?;
                        let n_ranges = r.read_u16::<LittleEndian>()? as usize;
                        let mut shard_ranges = Vec::with_capacity(n_ranges);
                        for _ in 0..n_ranges {
                            shard_ranges.push(ShardRange {
                                shard_id: r.read_u32::<LittleEndian>()?,
                                row_start: r.read_u32::<LittleEndian>()?,
                                row_end: r.read_u32::<LittleEndian>()?,
                            });
                        }
                        entries.push(CategoricalEntry {
                            value,
                            shard_ranges,
                        });
                    }
                    columns.push(IndexedColumn::Categorical(CategoricalIndex {
                        column_name,
                        entries,
                    }));
                }
                1 => {
                    // Numeric / B+ tree
                    let fanout = r.read_u16::<LittleEndian>()?;
                    let n_leaf_pages = r.read_u32::<LittleEndian>()? as usize;
                    let n_internal_pages = r.read_u32::<LittleEndian>()? as usize;
                    check_count_bound(n_leaf_pages, MAX_NUMERIC_PAGES, "leaf pages")?;
                    check_count_bound(n_internal_pages, MAX_NUMERIC_PAGES, "internal pages")?;

                    let mut internal_pages = Vec::with_capacity(capacity_hint(n_internal_pages));
                    for _ in 0..n_internal_pages {
                        let n_keys = r.read_u16::<LittleEndian>()?;
                        let mut keys = Vec::with_capacity(n_keys as usize);
                        for _ in 0..n_keys {
                            keys.push(r.read_f64::<LittleEndian>()?);
                        }
                        let mut children = Vec::with_capacity(n_keys as usize + 1);
                        for _ in 0..=n_keys {
                            children.push(r.read_u32::<LittleEndian>()?);
                        }
                        internal_pages.push(InternalPage {
                            n_keys,
                            keys,
                            children,
                        });
                    }

                    let mut leaf_pages = Vec::with_capacity(capacity_hint(n_leaf_pages));
                    for _ in 0..n_leaf_pages {
                        let n_leaf_entries = r.read_u32::<LittleEndian>()? as usize;
                        check_count_bound(
                            n_leaf_entries,
                            MAX_NUMERIC_LEAF_ENTRIES_PER_PAGE,
                            "leaf entries",
                        )?;
                        let mut entries = Vec::with_capacity(capacity_hint(n_leaf_entries));
                        for _ in 0..n_leaf_entries {
                            entries.push(NumericLeafEntry {
                                min_value: r.read_f64::<LittleEndian>()?,
                                max_value: r.read_f64::<LittleEndian>()?,
                                shard_id: r.read_u32::<LittleEndian>()?,
                                row_start: r.read_u32::<LittleEndian>()?,
                                row_end: r.read_u32::<LittleEndian>()?,
                            });
                        }
                        leaf_pages.push(LeafPage { entries });
                    }

                    columns.push(IndexedColumn::Numeric(NumericIndex {
                        column_name,
                        fanout,
                        internal_pages,
                        leaf_pages,
                    }));
                }
                _ => {
                    return Err(EngineError::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unknown predicate index column type: {column_type}"),
                    )));
                }
            }
        }

        Ok(PredicateIndex { version, columns })
    }

    /// Read the v2 (wide-counter) layout. See [`Self::write_to_v2`] for the
    /// wire format. The defensive `check_count_bound` / `capacity_hint`
    /// guards mirror the v1 path so the same allocation-DOS / malformed-
    /// input protections apply.
    fn read_from_v2<R: Read>(r: &mut R) -> Result<Self> {
        let version = 2u8;
        let n_columns = r.read_u32::<LittleEndian>()? as usize;
        // Bound the column count: v1 capped at u16::MAX (65535); even v2
        // realistic obs/var has <1000 columns, so a cap of 1M is generous.
        check_count_bound(n_columns, 1_000_000, "v2 columns")?;
        let mut columns = Vec::with_capacity(capacity_hint(n_columns));

        for _ in 0..n_columns {
            let name_len = r.read_u32::<LittleEndian>()? as usize;
            check_count_bound(name_len, 1_000_000, "v2 column_name length")?;
            let mut name_bytes = vec![0u8; name_len];
            r.read_exact(&mut name_bytes)?;
            let column_name = String::from_utf8(name_bytes).map_err(|e| {
                EngineError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })?;

            let column_type = r.read_u8()?;
            let _n_entries_u64 = r.read_u64::<LittleEndian>()?;

            match column_type {
                0 => {
                    let n_cat_entries = usize::try_from(_n_entries_u64).map_err(|_| {
                        EngineError::IoError(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "v2 categorical entries count exceeds system pointer width",
                        ))
                    })?;
                    check_count_bound(
                        n_cat_entries,
                        MAX_CATEGORICAL_ENTRIES,
                        "v2 categorical entries",
                    )?;
                    let mut entries = Vec::with_capacity(capacity_hint(n_cat_entries));
                    for _ in 0..n_cat_entries {
                        let val_len = r.read_u32::<LittleEndian>()? as usize;
                        check_count_bound(val_len, 1_000_000, "v2 categorical value length")?;
                        let mut val_bytes = vec![0u8; val_len];
                        r.read_exact(&mut val_bytes)?;
                        let value = String::from_utf8(val_bytes).map_err(|e| {
                            EngineError::IoError(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                e,
                            ))
                        })?;
                        let n_ranges = r.read_u32::<LittleEndian>()? as usize;
                        check_count_bound(n_ranges, 1_000_000_000, "v2 categorical shard_ranges")?;
                        let mut shard_ranges = Vec::with_capacity(capacity_hint(n_ranges));
                        for _ in 0..n_ranges {
                            shard_ranges.push(ShardRange {
                                shard_id: r.read_u32::<LittleEndian>()?,
                                row_start: r.read_u32::<LittleEndian>()?,
                                row_end: r.read_u32::<LittleEndian>()?,
                            });
                        }
                        entries.push(CategoricalEntry {
                            value,
                            shard_ranges,
                        });
                    }
                    columns.push(IndexedColumn::Categorical(CategoricalIndex {
                        column_name,
                        entries,
                    }));
                }
                1 => {
                    let fanout = r.read_u16::<LittleEndian>()?;
                    let n_leaf_pages =
                        usize::try_from(r.read_u64::<LittleEndian>()?).map_err(|_| {
                            EngineError::IoError(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "v2 leaf pages count exceeds system pointer width",
                            ))
                        })?;
                    let n_internal_pages =
                        usize::try_from(r.read_u64::<LittleEndian>()?).map_err(|_| {
                            EngineError::IoError(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "v2 internal pages count exceeds system pointer width",
                            ))
                        })?;
                    check_count_bound(n_leaf_pages, MAX_NUMERIC_PAGES, "v2 leaf pages")?;
                    check_count_bound(n_internal_pages, MAX_NUMERIC_PAGES, "v2 internal pages")?;

                    let mut internal_pages = Vec::with_capacity(capacity_hint(n_internal_pages));
                    for _ in 0..n_internal_pages {
                        let n_keys = r.read_u16::<LittleEndian>()?;
                        let mut keys = Vec::with_capacity(n_keys as usize);
                        for _ in 0..n_keys {
                            keys.push(r.read_f64::<LittleEndian>()?);
                        }
                        let mut children = Vec::with_capacity(n_keys as usize + 1);
                        for _ in 0..=n_keys {
                            children.push(r.read_u32::<LittleEndian>()?);
                        }
                        internal_pages.push(InternalPage {
                            n_keys,
                            keys,
                            children,
                        });
                    }

                    let mut leaf_pages = Vec::with_capacity(capacity_hint(n_leaf_pages));
                    for _ in 0..n_leaf_pages {
                        let n_leaf_entries = usize::try_from(r.read_u64::<LittleEndian>()?)
                            .map_err(|_| {
                                EngineError::IoError(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    "v2 leaf entries count exceeds system pointer width",
                                ))
                            })?;
                        check_count_bound(
                            n_leaf_entries,
                            MAX_NUMERIC_LEAF_ENTRIES_PER_PAGE,
                            "v2 leaf entries",
                        )?;
                        let mut entries = Vec::with_capacity(capacity_hint(n_leaf_entries));
                        for _ in 0..n_leaf_entries {
                            entries.push(NumericLeafEntry {
                                min_value: r.read_f64::<LittleEndian>()?,
                                max_value: r.read_f64::<LittleEndian>()?,
                                shard_id: r.read_u32::<LittleEndian>()?,
                                row_start: r.read_u32::<LittleEndian>()?,
                                row_end: r.read_u32::<LittleEndian>()?,
                            });
                        }
                        leaf_pages.push(LeafPage { entries });
                    }

                    columns.push(IndexedColumn::Numeric(NumericIndex {
                        column_name,
                        fanout,
                        internal_pages,
                        leaf_pages,
                    }));
                }
                _ => {
                    return Err(EngineError::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unknown v2 predicate index column type: {column_type}"),
                    )));
                }
            }
        }

        Ok(PredicateIndex { version, columns })
    }
}

// ============================================================================
// C3. Index construction
// ============================================================================

/// Build predicate indexes from metadata and shard row ranges.
///
/// `metadata` is either the obs or var RecordBatch.
/// `shard_row_ranges` contains `(row_start, row_end)` tuples per shard.
/// `indexed_columns` specifies which columns to index. If empty, auto-detect
/// columns with <1000 unique values.
pub fn build_indexes(
    metadata: &arrow::array::RecordBatch,
    shard_row_ranges: &[(u64, u64)],
    indexed_columns: &[String],
) -> Result<PredicateIndex> {
    let schema = metadata.schema();
    let mut columns_to_index: Vec<(String, usize)> = Vec::new();

    if indexed_columns.is_empty() {
        // Auto-detect: index columns with <1000 unique values
        for (i, field) in schema.fields().iter().enumerate() {
            let col = metadata.column(i);
            let n_unique = estimate_unique_values(col);
            if n_unique < 1000 {
                columns_to_index.push((field.name().clone(), i));
            }
        }
    } else {
        for name in indexed_columns {
            if let Some((i, _)) = schema.column_with_name(name) {
                let col = metadata.column(i);
                let n_unique = estimate_unique_values(col);
                // Skip columns with >10K unique values
                if n_unique <= 10_000 {
                    columns_to_index.push((name.clone(), i));
                }
            }
        }
    }

    let mut indexed = Vec::new();
    for (col_name, col_idx) in &columns_to_index {
        let col = metadata.column(*col_idx);
        let field = schema.field(*col_idx);
        let dt = field.data_type();

        if is_categorical_type(dt) {
            let cat_index = build_categorical_index(col, col_name, shard_row_ranges);
            indexed.push(IndexedColumn::Categorical(cat_index));
        } else if is_numeric_type(dt) {
            let num_index = build_numeric_index(col, col_name, shard_row_ranges, 64);
            indexed.push(IndexedColumn::Numeric(num_index));
        }
        // Skip unsupported types silently
    }

    Ok(PredicateIndex {
        version: 1,
        columns: indexed,
    })
}

// ============================================================================
// C3a. Conversion-time builder (Phase 5a)
// ============================================================================
//
// `build_indexes` above is the low-level entry point used by tests and
// rewrite-style callers that already know exactly which columns to index.
// The conversion pipeline needs a few more decisions surfaced:
//   - distinguish *forced* columns (`--index-obs` / `--index-var`) from
//     *preset* columns (`--index-preset`) so the caller can decide
//     whether a missing/unsupported column is fatal or just a warning,
//   - skip named columns whose cardinality exceeds a configurable cap
//     (rather than the silent 10_000 cap inside `build_indexes`),
//   - return `Option<Vec<u8>>` directly so the caller can pipe the bytes
//     to `ScxWriter::write_obs_predicate_index` without re-serialising.

/// Options for [`build_obs_predicate_index_bytes`] /
/// [`build_var_predicate_index_bytes`]. Constructed by the conversion
/// pipeline from CLI / Python flags.
#[derive(Debug, Clone, Default)]
pub struct PredicateIndexBuildOptions {
    /// Columns from `--index-obs` / `--index-var`. Missing or unsupported
    /// forced columns produce a [`BuildOutcome::ForcedColumnError`] which
    /// the caller surfaces as a hard error.
    pub forced_columns: Vec<String>,
    /// Columns from `--index-preset`. Missing produce
    /// [`BuildOutcome::PresetSkipped`] which the caller demotes to a
    /// `MissingPresetIndexColumn` warning.
    pub preset_columns: Vec<String>,
    /// Cardinality cap for auto-detection when both `forced` and `preset`
    /// are empty. Default 1000 (matches the implicit threshold in
    /// [`build_indexes`]).
    pub auto_threshold: usize,
    /// Hard cap above which a *named* (forced or preset) column is
    /// rejected as unsupported. Default 100_000.
    pub high_cardinality_threshold: usize,
}

/// Why a named column couldn't be indexed. Carried by both
/// [`BuildOutcome::ForcedColumnError`] and
/// [`BuildOutcome::PresetSkipped`] so callers can route policy
/// (e.g. `MissingPresetIndexColumn` vs `UnsupportedIndexColumn`)
/// without parsing a free-form `String`.
#[derive(Debug, Clone, PartialEq)]
pub enum SkipReason {
    /// Column name was not found in the metadata schema.
    MissingColumn,
    /// Column exists but its dtype is neither categorical nor numeric.
    UnsupportedDtype(String),
    /// Cardinality exceeds the caller-supplied
    /// `high_cardinality_threshold`.
    HighCardinality { n_unique: usize, threshold: usize },
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingColumn => write!(f, "missing column"),
            Self::UnsupportedDtype(dt) => write!(f, "unsupported dtype {dt}"),
            Self::HighCardinality {
                n_unique,
                threshold,
            } => write!(
                f,
                "cardinality {n_unique} exceeds high_cardinality_threshold {threshold}"
            ),
        }
    }
}

/// Per-column build outcome surfaced to the caller so policy decisions
/// (hard error vs. typed warning) stay in the conversion layer.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildOutcome {
    /// Forced column couldn't be indexed.
    ForcedColumnError { column: String, reason: SkipReason },
    /// Preset column couldn't be indexed.
    PresetSkipped { column: String, reason: SkipReason },
}

/// Render an actionable error message for a forced obs/var index
/// column that doesn't exist in the source DataFrame. Adds the
/// available column list and, when one is close enough, a single
/// `Did you mean '<col>'?` suggestion (Levenshtein-normalised
/// threshold ≥ 0.6). Shared by `scx-convert/src/pipeline.rs` (CLI
/// path) and `pyscx/src/convert/from_anndata.rs` (Python path) so both surfaces
/// emit the same message. — E2-2026-05-20.
///
/// Lives in `scx-engine` rather than `scx-convert` because pyscx
/// depends on `scx-engine` unconditionally but only pulls in
/// `scx-convert` under the `hdf5` feature; the helper must remain
/// callable from `pyscx::convert::build_and_write_predicate_indexes_inline`
/// (which is reachable from CPU-only `pyscx.from_anndata` paths).
pub fn forced_column_missing_message(axis: &str, column: &str, available: &[String]) -> String {
    format!(
        "forced {axis} index column '{column}': missing column. {}",
        column_suggestion_suffix(axis, column, available)
    )
}

/// Build the `"column not found. Available {axis} columns: [...]. Did you mean
/// '{...}'?"` reason for `EngineError::SchemaError` (the runtime predicate
/// path: `pyscx.open(...).query().filter_obs("totl_counts >= 500")`). The
/// suffix structure mirrors [`forced_column_missing_message`] so users see
/// the same "Available / Did you mean" treatment regardless of whether the
/// missing-column error originates from convert-time or query-time. —
/// F5-2026-05-20-Tier2.
pub fn column_not_found_message(axis: &str, column: &str, available: &[String]) -> String {
    format!(
        "column not found. {}",
        column_suggestion_suffix(axis, column, available)
    )
}

/// Find the closest match for `column` in `available` using normalized
/// Levenshtein distance with a 0.6 acceptance threshold. Returns `None`
/// when no candidate clears the threshold (i.e. the user's input isn't
/// a near-typo of any existing column).
fn best_match(column: &str, available: &[String]) -> Option<String> {
    available
        .iter()
        .map(|n| (n, strsim::normalized_levenshtein(column, n)))
        .filter(|(_, s)| *s >= 0.6)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(n, _)| n.clone())
}

/// Render the `"Available {axis} columns: [...]."` footer.
///
/// Lists every column when `available.len() <= 64`; the 64-column cap
/// keeps the worst-case message bounded for atlases with hundreds of
/// obs columns. Falls back to the first-8 preview with `", \u{2026}"`
/// (Unicode ellipsis — avoids the `, ....` 4-dot artifact) when there
/// are more than 64 columns.
///
/// N2-2026-05-21-Tier2: gating is purely on `available.len()` now.
/// PR #116's E1 fix originally gated show-all on "strsim has a winner",
/// but scanpy-vocab inputs (`total_counts`, `pct_counts_mt`, …) have
/// no near-match in the Census obs schema, so the user couldn't see
/// `raw_sum` in the truncated preview. Length is the only meaningful
/// concern; the strsim suggestion is rendered separately.
///
/// Empty `available` yields the empty-axis fallback that hints at no
/// obs/var metadata being present in the file.
fn render_available_columns(axis: &str, available: &[String]) -> String {
    if available.is_empty() {
        return format!("Available {axis} columns: [] (this h5ad has no {axis} metadata).");
    }
    if available.len() <= 64 {
        let items: Vec<&str> = available.iter().map(String::as_str).collect();
        return format!("Available {axis} columns: {items:?}.");
    }
    let preview_n = 8;
    let preview: Vec<&str> = available[..preview_n].iter().map(String::as_str).collect();
    format!("Available {axis} columns: {preview:?}, \u{2026}.")
}

/// Shared suffix builder for [`forced_column_missing_message`] and
/// [`column_not_found_message`]. Renders the available-columns footer
/// (capped at 64) and appends a strsim "Did you mean ...?" hint when
/// there's a near-match.
fn column_suggestion_suffix(axis: &str, column: &str, available: &[String]) -> String {
    let mut msg = render_available_columns(axis, available);
    if let Some(s) = best_match(column, available) {
        msg.push_str(&format!(" Did you mean '{s}'?"));
    }
    msg
}

/// Aggregate-form of [`forced_column_missing_message`] for callers that
/// processed multiple `BuildOutcome::ForcedColumnError` entries with
/// `SkipReason::MissingColumn`. Used by `scx convert --index-obs/--index-var`
/// (and the parallel `pyscx.from_anndata` path) so the user sees ALL
/// typos in a single error rather than fixing them one per run.
///
/// `missing` is the list of forced columns that came back missing, in
/// the order they were requested. Each entry gets its own line with a
/// strsim suggestion (if any); the available-columns preview is shared
/// across them via the same `column_suggestion_suffix` helper so the
/// rendered format stays consistent with the single-column message.
///
/// For a single missing column, prefer [`forced_column_missing_message`]
/// — the singular form keeps the existing single-line wording.
pub fn forced_columns_missing_message(
    axis: &str,
    missing: &[String],
    available: &[String],
) -> String {
    if missing.is_empty() {
        // Degenerate guard. Should not happen in practice — the caller
        // is supposed to gate on `!missing.is_empty()` before calling.
        return format!("0 forced {axis} index columns are missing.");
    }
    if missing.len() == 1 {
        return forced_column_missing_message(axis, &missing[0], available);
    }
    let mut msg = format!(
        "{n} forced {axis} index columns are missing:",
        n = missing.len()
    );
    for column in missing {
        match best_match(column, available) {
            Some(s) => msg.push_str(&format!("\n  - '{column}': did you mean '{s}'?")),
            None => msg.push_str(&format!("\n  - '{column}'")),
        }
    }
    msg.push('\n');
    msg.push_str(&render_available_columns(axis, available));
    msg
}

/// Named column preset (e.g. `cellxgene`, `perturbseq`, `training`).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexPreset {
    pub obs_columns: Vec<&'static str>,
    pub var_columns: Vec<&'static str>,
}

/// Resolve a named preset to its column list. Returns `None` for
/// unknown names; the conversion pipeline surfaces that as a clean
/// `ConvertError`.
pub fn index_preset_columns(name: &str) -> Option<IndexPreset> {
    match name {
        "cellxgene" => Some(IndexPreset {
            obs_columns: vec![
                "cell_type",
                "cell_type_ontology_term_id",
                "tissue",
                "tissue_ontology_term_id",
                "disease",
                "assay",
                "donor_id",
                "development_stage",
                "sex",
                "suspension_type",
            ],
            var_columns: vec!["feature_name", "feature_type"],
        }),
        "perturbseq" => Some(IndexPreset {
            obs_columns: vec![
                "cell_type",
                "donor",
                "batch",
                "condition",
                "perturbation",
                "guide_id",
                "target_gene",
                "control",
                "split",
            ],
            var_columns: vec!["feature_name", "feature_type"],
        }),
        "training" => Some(IndexPreset {
            obs_columns: vec![
                "cell_type",
                "donor",
                "batch",
                "dataset_id",
                "split",
                "organism",
                "tissue",
            ],
            var_columns: vec!["feature_name", "feature_type"],
        }),
        _ => None,
    }
}

/// Whether a named index preset implies a CSC sidecar should be built
/// when the caller did not explicitly pass a `csc` policy.
///
/// The accel-ready presets — `training` and `perturbseq` — drive
/// column/DE-heavy workloads (pseudobulk, `pdex_ref`, `rank_genes_groups`)
/// whose primary substrate is the column-major CSC sidecar, so selecting
/// one of those presets upgrades an *unset* `csc` to `auto`. `cellxgene`
/// is query/browse-oriented and does not imply CSC. An explicit `--csc`
/// value (including `off`) always wins over this default.
pub fn preset_implies_csc_auto(name: &str) -> bool {
    matches!(name, "training" | "perturbseq")
}

/// Resolve the effective CSC policy string for a conversion entry point.
///
/// An explicit `csc` always wins; when unset (`None`), an accel-ready
/// `index_preset` (`training` / `perturbseq`) upgrades the default to
/// `"auto"`, otherwise the default is `"off"`. Shared by the `scx convert`
/// CLI and the pyscx conversion entry points so the two front-ends cannot
/// drift.
pub fn resolve_csc_policy<'a>(csc: Option<&'a str>, index_preset: Option<&str>) -> Cow<'a, str> {
    match csc {
        Some(v) => Cow::Borrowed(v),
        None if index_preset.is_some_and(preset_implies_csc_auto) => Cow::Borrowed("auto"),
        None => Cow::Borrowed("off"),
    }
}

/// Build a predicate index for a metadata RecordBatch (obs or var) and
/// return the serialized bytes ready to pass to
/// `ScxWriter::write_obs_predicate_index` /
/// `write_var_predicate_index`.
///
/// `outcomes` accumulates per-column policy events (forced errors,
/// preset skips); the caller decides whether to fail or warn.
///
/// When `forced_columns` and `preset_columns` are both empty, this
/// falls back to auto-detection using `auto_threshold` as the maximum
/// cardinality for categoricals.
fn build_predicate_index_bytes_inner(
    metadata: &arrow::array::RecordBatch,
    shard_row_ranges: &[(u64, u64)],
    options: &PredicateIndexBuildOptions,
    outcomes: &mut Vec<BuildOutcome>,
    indexed_column_names: &mut Vec<String>,
) -> Result<Option<Vec<u8>>> {
    use std::collections::BTreeSet;

    let schema = metadata.schema();

    // Auto-detect path: when both forced and preset are empty, use the
    // existing `build_indexes` behaviour (cardinality < auto_threshold,
    // silent skip otherwise — no outcomes produced).
    if options.forced_columns.is_empty() && options.preset_columns.is_empty() {
        let mut columns_to_index: Vec<(String, usize)> = Vec::new();
        for (i, field) in schema.fields().iter().enumerate() {
            let col = metadata.column(i);
            let dt = field.data_type();
            if !is_categorical_type(dt) && !is_numeric_type(dt) {
                continue;
            }
            let n_unique = estimate_unique_values(col);
            if n_unique < options.auto_threshold {
                columns_to_index.push((field.name().clone(), i));
            }
        }
        if columns_to_index.is_empty() {
            return Ok(None);
        }
        let indexed: Vec<IndexedColumn> = columns_to_index
            .iter()
            .filter_map(|(name, idx)| {
                let col = metadata.column(*idx);
                let dt = schema.field(*idx).data_type();
                if is_categorical_type(dt) {
                    Some(IndexedColumn::Categorical(build_categorical_index(
                        col,
                        name,
                        shard_row_ranges,
                    )))
                } else if is_numeric_type(dt) {
                    Some(IndexedColumn::Numeric(build_numeric_index(
                        col,
                        name,
                        shard_row_ranges,
                        64,
                    )))
                } else {
                    None
                }
            })
            .collect();
        for (name, _) in &columns_to_index {
            indexed_column_names.push(name.clone());
        }
        let index = PredicateIndex {
            version: 1,
            columns: indexed,
        };
        let mut buf = Vec::new();
        index.write_to(&mut buf)?;
        return Ok(Some(buf));
    }

    // Named-column path: validate each forced / preset column and emit
    // outcomes for the ones that can't be indexed. Dedup forced ∪ preset;
    // forced wins (a column listed in both is treated as forced).
    let mut named: Vec<(String, bool /* is_forced */)> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for col in &options.forced_columns {
        if seen.insert(col.clone()) {
            named.push((col.clone(), true));
        }
    }
    for col in &options.preset_columns {
        if seen.insert(col.clone()) {
            named.push((col.clone(), false));
        }
    }

    let mut indexed: Vec<IndexedColumn> = Vec::new();
    let push_outcome =
        |outcomes: &mut Vec<BuildOutcome>, col_name: &str, is_forced: bool, reason: SkipReason| {
            outcomes.push(if is_forced {
                BuildOutcome::ForcedColumnError {
                    column: col_name.to_string(),
                    reason,
                }
            } else {
                BuildOutcome::PresetSkipped {
                    column: col_name.to_string(),
                    reason,
                }
            });
        };
    for (col_name, is_forced) in &named {
        let Some((idx, field)) = schema.column_with_name(col_name) else {
            push_outcome(outcomes, col_name, *is_forced, SkipReason::MissingColumn);
            continue;
        };
        let col = metadata.column(idx);
        let dt = field.data_type();
        if !is_categorical_type(dt) && !is_numeric_type(dt) {
            push_outcome(
                outcomes,
                col_name,
                *is_forced,
                SkipReason::UnsupportedDtype(format!("{dt:?}")),
            );
            continue;
        }
        let n_unique = estimate_unique_values(col);
        if n_unique > options.high_cardinality_threshold {
            push_outcome(
                outcomes,
                col_name,
                *is_forced,
                SkipReason::HighCardinality {
                    n_unique,
                    threshold: options.high_cardinality_threshold,
                },
            );
            continue;
        }
        if is_categorical_type(dt) {
            indexed.push(IndexedColumn::Categorical(build_categorical_index(
                col,
                col_name,
                shard_row_ranges,
            )));
        } else {
            indexed.push(IndexedColumn::Numeric(build_numeric_index(
                col,
                col_name,
                shard_row_ranges,
                64,
            )));
        }
        indexed_column_names.push(col_name.clone());
    }

    if indexed.is_empty() {
        return Ok(None);
    }
    let index = PredicateIndex {
        version: 1,
        columns: indexed,
    };
    let mut buf = Vec::new();
    index.write_to(&mut buf)?;
    Ok(Some(buf))
}

/// Build serialized predicate-index bytes for an obs RecordBatch.
///
/// `indexed_column_names` is populated with the names of the columns
/// that ended up in the index (useful for provenance stamping).
pub fn build_obs_predicate_index_bytes(
    obs: &arrow::array::RecordBatch,
    shard_row_ranges: &[(u64, u64)],
    options: &PredicateIndexBuildOptions,
    outcomes: &mut Vec<BuildOutcome>,
    indexed_column_names: &mut Vec<String>,
) -> Result<Option<Vec<u8>>> {
    build_predicate_index_bytes_inner(
        obs,
        shard_row_ranges,
        options,
        outcomes,
        indexed_column_names,
    )
}

// ============================================================================
// Streaming obs predicate-index builder
// ============================================================================

/// Streaming counterpart to [`build_obs_predicate_index_bytes`].
///
/// The single-batch entry point requires the full obs `RecordBatch` to
/// be assembled in memory before predicate-index construction starts,
/// which defeats the purpose of the row-sharded obs layout introduced
/// for atlas-scale merges and appends.
/// The builder accepts shards one at a time via [`Self::push_shard`]
/// and finalises into the same serialised bytes blob at [`Self::finish`].
///
/// Memory per indexed column: categorical accumulates
/// `BTreeMap<String, Vec<u64>>` indexed by value, as the batch-mode path
/// does; numeric accumulates one [`NumericSpan`] per
/// [`NUMERIC_BLOCK_ROWS`]-row block rather than a `(value, row)` pair per row,
/// which is what keeps an atlas-scale numeric column in the low megabytes
/// instead of 16 B per row per column. On top of that we never have to
/// materialise the obs `RecordBatch` itself — Arrow column buffers stay
/// shard-local and are released after each `push_shard`.
///
/// **Push one shard per call.** The numeric accumulator summarises within a
/// push, so pushing a batch that spans several of the shard ranges later
/// handed to [`Self::finish`] gives each of them the same widened bounds —
/// sound, but it costs the Level-1 pruning the index is for. Very few callers
/// naturally push at that granularity: they chunk obs by `shard_target_rows`
/// (`modify_metadata`, merge's concatenating path, append's new rows) or push
/// a whole axis at once (the batch conversion entry point, append's
/// convert-on-append path), while `finish` is given the **CSR** shard ranges,
/// which need not match either. Use [`Self::push_shard_split`] with the ranges
/// you will pass to `finish`; it is a precision hint, so an approximation is
/// still worth passing.
///
/// Column selection works identically to the batch-mode path:
/// - forced + preset columns are tracked from the start (even if they
///   exceed the auto-detect threshold — `high_cardinality_threshold`
///   still caps them with a `HighCardinality` outcome at finish);
/// - auto-detect mode tracks every supported column up front and at
///   finish keeps only those whose observed cardinality stays under
///   `auto_threshold`.
pub struct ObsPredicateIndexBuilder {
    schema: arrow::datatypes::SchemaRef,
    options: PredicateIndexBuildOptions,
    /// Columns we are actively accumulating. Sorted in the order they
    /// will eventually appear in the serialised `PredicateIndex`.
    columns: Vec<ColumnAccumulator>,
    /// Total rows pushed so far. Used to verify the shard stream covers
    /// the obs axis the caller claimed.
    rows_pushed: u64,
}

struct ColumnAccumulator {
    name: String,
    /// `true` when the column was named via forced/preset config (so
    /// failures map to `ForcedColumnError` / `PresetSkipped`); `false`
    /// for auto-detected columns (silent skip).
    is_forced: bool,
    /// `true` when the column was named via forced/preset config
    /// (vs. auto-detect). Forced columns surface a `ForcedColumnError`
    /// on skip; auto-detected columns are skipped silently.
    is_named: bool,
    /// Auto-detect or named-cap mode? Drives the cardinality check at
    /// finish.
    selection: ColumnSelection,
    /// Per-shard payload state. Determined at `new` from the column's
    /// declared dtype in the schema.
    state: ColumnState,
}

#[derive(Debug, Clone, Copy)]
enum ColumnSelection {
    /// Auto-detect: include at finish iff observed cardinality stays
    /// under `options.auto_threshold`.
    Auto,
    /// Named (forced or preset): include unless observed cardinality
    /// exceeds `options.high_cardinality_threshold`.
    Named,
}

/// Rows per streaming numeric summary block, counted from the start of each
/// push.
///
/// The shard partition is not known until `finish` — it is *not* inherently
/// the `push_shard` partition, and cannot be moved earlier: `scx-ops`' merge
/// builds its output shard ranges while this builder is already accumulating.
/// So the accumulator cannot summarise per shard directly; it summarises per
/// fixed-size block of rows *within each push* and folds those onto shards at
/// `finish`.
///
/// Blocks never span two pushes, which is what makes the block size
/// irrelevant to precision for a caller that pushes one shard at a time: all
/// of a push's blocks then lie inside one shard, and the fold takes min/max
/// across them, so the result is exact at any shard size. Every writer whose
/// pushes can straddle reaches that state via
/// [`ObsPredicateIndexBuilder::push_shard_split`]. Merge's sorted path is the
/// one that still calls `push_shard` directly, and correctly: it flushes obs
/// and X at the same `shard_target_rows`, so its pushes already are the shard
/// partition.
///
/// A block can still straddle where a caller's range hint does not match what
/// `finish` receives. For that case the block size sets both the memory bound
/// (one
/// [`NumericSpan`], 32 B, per block per column — ~1.5 MB for a 50M-row axis,
/// against 800 MB for the whole-axis `(value, row)` list this replaced) and
/// the bound on how far a straddling block can widen a shard's recorded
/// `[min, max]` (at most the values of the ≤ `NUMERIC_BLOCK_ROWS - 1` rows on
/// the far side of the boundary). Widening only ever costs pruning power,
/// never correctness — see [`numeric_leaves_from_spans`].
const NUMERIC_BLOCK_ROWS: u64 = 1024;

enum ColumnState {
    Categorical {
        values: BTreeMap<String, Vec<u64>>,
    },
    /// One [`NumericSpan`] per `NUMERIC_BLOCK_ROWS`-row block that carried at
    /// least one value, in ascending row order (`push_shard` enforces that
    /// rows arrive contiguously and in order).
    Numeric {
        blocks: Vec<NumericSpan>,
    },
    Unsupported {
        dtype: String,
    },
    MissingColumn,
}

impl ObsPredicateIndexBuilder {
    /// Construct a builder by inspecting the obs schema and the
    /// forced/preset column list. Auto-detect mode (no forced /
    /// preset given) tracks every categorical / numeric column up front
    /// and resolves cardinality at finish.
    pub fn new(
        schema: arrow::datatypes::SchemaRef,
        options: &PredicateIndexBuildOptions,
    ) -> Result<Self> {
        use std::collections::BTreeSet;

        let auto_mode = options.forced_columns.is_empty() && options.preset_columns.is_empty();
        let mut columns: Vec<ColumnAccumulator> = Vec::new();

        if auto_mode {
            for field in schema.fields().iter() {
                let dt = field.data_type();
                if !is_categorical_type(dt) && !is_numeric_type(dt) {
                    continue;
                }
                let state = if is_categorical_type(dt) {
                    ColumnState::Categorical {
                        values: BTreeMap::new(),
                    }
                } else {
                    ColumnState::Numeric { blocks: Vec::new() }
                };
                columns.push(ColumnAccumulator {
                    name: field.name().clone(),
                    is_forced: false,
                    is_named: false,
                    selection: ColumnSelection::Auto,
                    state,
                });
            }
        } else {
            // Named-column path: walk forced ∪ preset (forced wins
            // on duplicates), validate each against the schema, and
            // record per-column state — including unsupported and
            // missing markers so `finish` can emit per-column
            // outcomes without re-walking the schema.
            let mut seen: BTreeSet<String> = BTreeSet::new();
            let mut named: Vec<(String, bool)> = Vec::new();
            for col in &options.forced_columns {
                if seen.insert(col.clone()) {
                    named.push((col.clone(), true));
                }
            }
            for col in &options.preset_columns {
                if seen.insert(col.clone()) {
                    named.push((col.clone(), false));
                }
            }
            for (col_name, is_forced) in named {
                let (state, _is_supported) = match schema.column_with_name(&col_name) {
                    Some((_, field)) => {
                        let dt = field.data_type();
                        if is_categorical_type(dt) {
                            (
                                ColumnState::Categorical {
                                    values: BTreeMap::new(),
                                },
                                true,
                            )
                        } else if is_numeric_type(dt) {
                            (ColumnState::Numeric { blocks: Vec::new() }, true)
                        } else {
                            (
                                ColumnState::Unsupported {
                                    dtype: format!("{dt:?}"),
                                },
                                false,
                            )
                        }
                    }
                    None => (ColumnState::MissingColumn, false),
                };
                columns.push(ColumnAccumulator {
                    name: col_name,
                    is_forced,
                    is_named: true,
                    selection: ColumnSelection::Named,
                    state,
                });
            }
        }

        Ok(Self {
            schema,
            options: options.clone(),
            columns,
            rows_pushed: 0,
        })
    }

    /// Accumulate one obs metadata shard. `shard_row_offset` is the
    /// global row index where this shard begins (i.e. its
    /// `row_start`); the builder pairs each row value with
    /// `shard_row_offset + i` so the final `rows_to_shard_ranges` pass
    /// at `finish` can map values back to per-shard ranges identical
    /// to what the batch-mode builder would produce on the assembled
    /// `RecordBatch`.
    ///
    /// Shards must be pushed in row order and must form a contiguous
    /// cover (no gap, no overlap) of the obs axis. The builder
    /// verifies the order against [`Self::rows_pushed`] but trusts
    /// the caller's `shard_row_offset` for correctness.
    ///
    /// A `batch` should not span more than one of the shard ranges passed to
    /// [`Self::finish`] — nothing enforces it, and the categorical side is
    /// insensitive to it, but the numeric side summarises per push and will
    /// hand every spanned shard the same widened bounds. Prefer
    /// [`Self::push_shard_split`] wherever the shard ranges are known. See the
    /// type-level docs.
    pub fn push_shard(
        &mut self,
        batch: &arrow::array::RecordBatch,
        shard_row_offset: u64,
    ) -> Result<()> {
        // Sanity check: shards must be pushed in row order.
        if shard_row_offset != self.rows_pushed {
            return Err(EngineError::Generic(format!(
                "ObsPredicateIndexBuilder: shard at row offset {shard_row_offset} \
                 does not continue from previous cumulative rows {prev}",
                prev = self.rows_pushed
            )));
        }

        // Verify the shard's schema matches what `new()` was given.
        // We compare a normalised "class" (categorical / numeric /
        // other) rather than exact dtypes so a builder initialised
        // with `Utf8` cell ids accepts shards that present them as
        // `LargeUtf8` (which is what the per-shard upcast in
        // `ScxWriter::write_arrow_ipc` emits for >2 GB string
        // payloads). `extract_string_value` / `extract_numeric_value`
        // handle both widths transparently.
        let shard_schema = batch.schema();
        for col in &self.columns {
            let (Some((_, shard_field)), Some((_, expected_field))) = (
                shard_schema.column_with_name(&col.name),
                self.schema.column_with_name(&col.name),
            ) else {
                continue;
            };
            let same_class =
                column_class_compatible(shard_field.data_type(), expected_field.data_type());
            if !same_class {
                return Err(EngineError::Generic(format!(
                    "ObsPredicateIndexBuilder: shard column '{}' has dtype \
                     {:?}, builder was initialised with {:?} — these are \
                     not interchangeable for predicate-index purposes",
                    col.name,
                    shard_field.data_type(),
                    expected_field.data_type()
                )));
            }
        }

        let n_rows = batch.num_rows();
        for col in self.columns.iter_mut() {
            // Skip columns that already errored (missing, unsupported);
            // their outcome is emitted at `finish`.
            let (col_idx, _) = match shard_schema.column_with_name(&col.name) {
                Some(x) => x,
                None => continue,
            };
            let array = batch.column(col_idx);
            match &mut col.state {
                ColumnState::Categorical { values } => {
                    for i in 0..n_rows {
                        if array.is_null(i) {
                            continue;
                        }
                        if let Some(v) = extract_string_value(array, i) {
                            values
                                .entry(v)
                                .or_default()
                                .push(shard_row_offset + i as u64);
                        }
                    }
                }
                ColumnState::Numeric { blocks } => {
                    // Fold into the open block, opening a new one whenever the
                    // row crosses a block boundary. Blocks are counted from
                    // the start of *this push*, never across pushes, so a
                    // push confined to one shard yields shard-local blocks and
                    // exact bounds however small the shards are. Callers get
                    // there by calling `push_shard_split` rather than by
                    // pushing at CSR granularity naturally — almost none of
                    // them does; see its rustdoc for which and why.
                    let mut open_block: Option<u64> = None;
                    for i in 0..n_rows {
                        if array.is_null(i) {
                            continue;
                        }
                        let Some(v) = extract_numeric_value(array, i) else {
                            continue;
                        };
                        let row = shard_row_offset + i as u64;
                        let block = i as u64 / NUMERIC_BLOCK_ROWS;
                        match blocks.last_mut() {
                            Some(open) if open_block == Some(block) => {
                                open.min = open.min.min(v);
                                open.max = open.max.max(v);
                                open.last_row = row;
                            }
                            _ => {
                                blocks.push(NumericSpan {
                                    min: v,
                                    max: v,
                                    first_row: row,
                                    last_row: row,
                                });
                                open_block = Some(block);
                            }
                        }
                    }
                }
                ColumnState::Unsupported { .. } | ColumnState::MissingColumn => {}
            }
        }

        // Defense-in-depth: a u64 cumulative-row counter can't realistically
        // overflow (would require > 2^64 rows pushed across a single
        // builder invocation), but use `checked_add` so a malformed shard
        // claiming `n_rows >= 2^64 - rows_pushed` fails loudly instead of
        // silently saturating and corrupting the downstream row-offset
        // arithmetic.
        self.rows_pushed = self.rows_pushed.checked_add(n_rows as u64).ok_or_else(|| {
            EngineError::Generic(format!(
                "ObsPredicateIndexBuilder: cumulative row count overflowed u64 \
                     pushing shard at offset {shard_row_offset} with {n_rows} rows",
            ))
        })?;
        Ok(())
    }

    /// [`Self::push_shard`], but split wherever a `shard_row_ranges` boundary
    /// falls inside `batch` — one push per shard the batch covers.
    ///
    /// Use this instead of `push_shard` whenever the caller knows the ranges
    /// it will hand [`Self::finish`], which is most of them. `push_shard`
    /// means what it says, and callers do not all honour it: the batch
    /// conversion entry point pushes the whole obs axis in one call,
    /// `compact` / `sort` push *input* shards against *output* ranges a
    /// reshape has moved, append's convert-on-append path pushes the entire
    /// pre-append axis, and `modify_metadata` / merge chunk obs by
    /// `shard_target_rows` while finishing over CSR shards that need not
    /// match. Each of those hands every spanned shard the same widened
    /// `[min, max]`, which cannot produce a wrong answer but does erase the
    /// Level-1 pruning the index exists to provide.
    ///
    /// `shard_row_ranges` is a **precision hint, not a correctness input**:
    /// it only decides where pushes are cut. Ranges that turn out not to
    /// match what `finish` receives cost some pruning power and nothing else,
    /// so a caller that can only approximate them should still pass them.
    ///
    /// `RecordBatch::slice` shares Arrow buffers, so each piece is O(1).
    pub fn push_shard_split(
        &mut self,
        batch: &arrow::array::RecordBatch,
        row_offset: u64,
        shard_row_ranges: &[(u64, u64)],
    ) -> Result<()> {
        let n_rows = batch.num_rows() as u64;
        if n_rows == 0 {
            return self.push_shard(batch, row_offset);
        }
        let end = row_offset + n_rows;

        // Only the ranges overlapping this batch can contribute a cut. When the
        // table is well-formed and ascending they are a contiguous window, so
        // find its start and stop at its end instead of visiting every range
        // on every push: callers push roughly one batch per shard, which made
        // the full scan quadratic in shard count before the leaf fold even ran.
        let mut cuts: Vec<u64> = if ranges_ascending_disjoint(shard_row_ranges) {
            let first = shard_row_ranges.partition_point(|&(_, stop)| stop <= row_offset);
            // Ascending and disjoint ⇒ the emitted boundaries come out sorted.
            shard_row_ranges[first..]
                .iter()
                .take_while(|&&(start, _)| start < end)
                .flat_map(|&(start, stop)| [start, stop])
                .filter(|&b| b > row_offset && b < end)
                .collect()
        } else {
            let mut all: Vec<u64> = shard_row_ranges
                .iter()
                .flat_map(|&(start, stop)| [start, stop])
                .filter(|&b| b > row_offset && b < end)
                .collect();
            all.sort_unstable();
            all
        };
        cuts.dedup();

        let mut cursor = row_offset;
        for cut in cuts.into_iter().chain(std::iter::once(end)) {
            self.push_shard(
                &batch.slice((cursor - row_offset) as usize, (cut - cursor) as usize),
                cursor,
            )?;
            cursor = cut;
        }
        Ok(())
    }

    /// Finalise the index: resolve auto-detect cardinality, build
    /// `CategoricalIndex` / `NumericIndex` entries, serialise to the
    /// `PredicateIndex` byte format, and return the resulting bytes.
    ///
    /// Outcomes (forced errors, preset skips, high-cardinality skips,
    /// missing-column markers) are pushed onto `outcomes` so the
    /// caller can demote them to typed warnings (same protocol as
    /// [`build_obs_predicate_index_bytes`]). `indexed_column_names`
    /// receives the names of the columns that ended up in the index,
    /// in serialisation order — used for provenance stamping.
    pub fn finish(
        self,
        shard_row_ranges: &[(u64, u64)],
        outcomes: &mut Vec<BuildOutcome>,
        indexed_column_names: &mut Vec<String>,
    ) -> Result<Option<Vec<u8>>> {
        let push_outcome = |outcomes: &mut Vec<BuildOutcome>,
                            col_name: &str,
                            is_forced: bool,
                            reason: SkipReason| {
            outcomes.push(if is_forced {
                BuildOutcome::ForcedColumnError {
                    column: col_name.to_string(),
                    reason,
                }
            } else {
                BuildOutcome::PresetSkipped {
                    column: col_name.to_string(),
                    reason,
                }
            });
        };

        let mut indexed: Vec<IndexedColumn> = Vec::new();
        let mut auto_mode_index_count = 0usize;

        for col in self.columns {
            let ColumnAccumulator {
                name,
                is_forced,
                is_named,
                selection,
                state,
            } = col;
            match state {
                ColumnState::MissingColumn => {
                    if is_named {
                        push_outcome(outcomes, &name, is_forced, SkipReason::MissingColumn);
                    }
                    continue;
                }
                ColumnState::Unsupported { dtype } => {
                    if is_named {
                        push_outcome(
                            outcomes,
                            &name,
                            is_forced,
                            SkipReason::UnsupportedDtype(dtype),
                        );
                    }
                    continue;
                }
                ColumnState::Categorical { values } => {
                    let n_unique = values.len();
                    match selection {
                        ColumnSelection::Auto => {
                            // Silent skip when cardinality exceeds the
                            // auto-detect threshold.
                            if n_unique >= self.options.auto_threshold {
                                continue;
                            }
                        }
                        ColumnSelection::Named => {
                            if n_unique > self.options.high_cardinality_threshold {
                                push_outcome(
                                    outcomes,
                                    &name,
                                    is_forced,
                                    SkipReason::HighCardinality {
                                        n_unique,
                                        threshold: self.options.high_cardinality_threshold,
                                    },
                                );
                                continue;
                            }
                        }
                    }

                    let mut entries: Vec<CategoricalEntry> = Vec::new();
                    for (value, rows) in &values {
                        let shard_ranges = rows_to_shard_ranges(rows, shard_row_ranges);
                        entries.push(CategoricalEntry {
                            value: value.clone(),
                            shard_ranges,
                        });
                    }
                    indexed.push(IndexedColumn::Categorical(CategoricalIndex {
                        column_name: name.clone(),
                        entries,
                    }));
                    indexed_column_names.push(name);
                    if matches!(selection, ColumnSelection::Auto) {
                        auto_mode_index_count += 1;
                    }
                }
                ColumnState::Numeric { blocks } => {
                    // Intentional no-op: numeric columns always build a
                    // B+ tree regardless of cardinality, so unlike the
                    // categorical path there's no "skip on high
                    // cardinality" branch. The condition is kept (as
                    // a `let _` below) only so future readers see that
                    // we considered cardinality and chose to ignore it.
                    //
                    // There is nothing left for a cap to bound: the index is
                    // one leaf entry per shard whatever the column holds, so
                    // a million distinct values and ten cost the same bytes.
                    // (`blocks.len()` isn't a cardinality either — it is the
                    // number of `NUMERIC_BLOCK_ROWS`-row blocks that carried
                    // a value — so a check here would be doubly misleading.)
                    //
                    // NOTE: this still does **not** match batch-mode
                    // behaviour. `build_predicate_index_bytes_inner` applies
                    // `high_cardinality_threshold` to every column before
                    // the categorical/numeric split, so a high-cardinality
                    // numeric column is skipped there and indexed here. What
                    // survives of that divergence is not an index-size
                    // argument any more — it is that the batch path drops the
                    // column outright, which changes index presence,
                    // `indexed_column_names`, the serialised bytes and the
                    // Level-1 pruning available on it, while this path keeps
                    // it for a handful of bytes. Unifying them would change
                    // batch-path behaviour, so it is left alone and documented
                    // (docs/api.md) rather than changed here.
                    //
                    // It is also not a divergence any conversion front end can
                    // reach on obs: `scx convert` and `pyscx.from_anndata` go
                    // through `build_and_write_conversion_predicate_indexes`,
                    // which hands obs to *this* builder via a one-item
                    // iterator. The batch path owns `var`, and obs only for
                    // direct callers of `build_obs_predicate_index_bytes`.
                    let _ = matches!(selection, ColumnSelection::Named)
                        && blocks.len() > self.options.high_cardinality_threshold;
                    indexed.push(IndexedColumn::Numeric(numeric_index_from_spans(
                        &name,
                        blocks.into_iter(),
                        shard_row_ranges,
                        64,
                    )));
                    indexed_column_names.push(name);
                    if matches!(selection, ColumnSelection::Auto) {
                        auto_mode_index_count += 1;
                    }
                }
            }
        }

        // Empty index: nothing was selected (either no auto-detect
        // candidates passed the threshold, or every named column was
        // skipped). Match batch-mode semantics — return None so the
        // caller skips the `write_*_predicate_index` call.
        if indexed.is_empty() {
            return Ok(None);
        }
        // Cosmetic guard: auto-detect mode produced at least one
        // entry. Keeps the invariant identical to the batch path
        // where `columns_to_index.is_empty()` short-circuits.
        let _ = auto_mode_index_count;

        let index = PredicateIndex {
            version: 1,
            columns: indexed,
        };
        let mut buf = Vec::new();
        index.write_to(&mut buf)?;
        Ok(Some(buf))
    }
}

/// Are the shard ranges each well-formed *and* ascending and disjoint?
///
/// Both halves matter. Checking only `end <= next.start` accepts an inverted
/// range like `[(100, 50), (60, 70)]`, and the scans below would then stop at
/// the inverted entry's `start` and miss the valid overlap behind it — the
/// exhaustive fallback exists precisely so a malformed table cannot lose an
/// overlap, so the predicate that selects it has to be the stronger one.
/// Every in-tree table is catalog-derived and valid; this is about the public
/// helper's stated contract, not about a reachable in-tree bug.
fn ranges_ascending_disjoint(ranges: &[(u64, u64)]) -> bool {
    ranges.iter().all(|&(start, stop)| start <= stop) && ranges.windows(2).all(|w| w[0].1 <= w[1].0)
}

/// One conservative summary of a contiguous run of obs rows: `[min, max]`
/// bounds every valued row in `[first_row, last_row]`, in **global** row
/// space. Rows in between that carry no value simply do not narrow it.
///
/// This is the single input shape both numeric builders reduce to. The batch
/// builder emits one span per valued row (so it stays exact); the streaming
/// builder emits one per fixed-size block of rows (so its memory is bounded).
#[derive(Debug, Clone, Copy)]
struct NumericSpan {
    min: f64,
    max: f64,
    first_row: u64,
    last_row: u64,
}

/// Fold row-ordered spans into **at most one [`NumericLeafEntry`] per shard** —
/// a shard no valued span reaches emits none, which is what keeps a null tail
/// out of the coverage calculation.
///
/// Per-shard is the granularity the leaves are actually consumed at:
/// [`derive_shard_column_stats`] folds them to a per-shard
/// `ColumnStat::MinMax` (what Level-1 pruning reads) and
/// [`PredicateIndex::max_covered_global_row`] takes the per-shard maximum
/// `row_end` (what [`index_covers_all_obs`] reads). No query path ever looks
/// at an individual leaf — `eval_rowset` is residual for every numeric
/// operator and there is no range-lookup method — so emitting finer leaves
/// only costs bytes. Before this was per-shard, a continuous obs column
/// produced ~1 leaf per row (28 B each on the wire): ~1.4 GB for a single
/// indexed numeric column on a 50M-cell atlas.
///
/// A span that straddles a shard boundary contributes its bounds to **both**
/// shards. That widening is sound in the only direction that matters:
/// Level-1 pruning excludes a shard when the probe falls outside
/// `[min, max]`, so a wider bound can only fail to prune, never prune a
/// shard that holds a match.
///
/// The row range is clamped to `[first_row, last_row + 1]` rather than to the
/// span's full extent, which is what keeps **coverage** exact. A span whose
/// valued rows all sit on one side of a boundary yields `lo >= hi` on the
/// other side and contributes nothing there, so `max_covered_global_row` can
/// never claim a row that carried no value. Under-claiming coverage is safe
/// (the query falls back to a full obs scan); over-claiming would let a
/// stale post-`append` index be trusted.
///
/// `spans` must arrive in ascending row order — both builders produce them
/// that way. Out-of-order spans still fold correctly (the cursor rewinds),
/// just more slowly.
fn numeric_leaves_from_spans(
    spans: impl Iterator<Item = NumericSpan>,
    shard_row_ranges: &[(u64, u64)],
) -> Vec<NumericLeafEntry> {
    let mut acc: Vec<Option<NumericLeafEntry>> = vec![None; shard_row_ranges.len()];

    // Every in-tree range table is ascending and disjoint, but nothing in the
    // signature promises it — `global_row_to_shard` scans exhaustively for
    // exactly that reason. Check once instead of assuming: when it holds, the
    // window of shards a span can overlap only moves forward (spans arrive in
    // row order), so a cursor makes the fold O(spans + shards) rather than
    // O(spans × shards). The batch builder emits one span per row, so that
    // product is the whole column times the shard count.
    let ascending_disjoint = ranges_ascending_disjoint(shard_row_ranges);
    let mut cursor = 0usize;

    for span in spans {
        let first = if ascending_disjoint {
            // Row order is a documented precondition, not an enforced one. A
            // span that goes backwards rewinds the cursor, costing a rescan
            // rather than a wrong answer.
            if cursor > 0 && shard_row_ranges[cursor - 1].1 > span.first_row {
                cursor = 0;
            }
            while cursor < shard_row_ranges.len() && shard_row_ranges[cursor].1 <= span.first_row {
                cursor += 1;
            }
            cursor
        } else {
            0
        };

        for (i, &(s_start, s_end)) in shard_row_ranges.iter().enumerate().skip(first) {
            if ascending_disjoint && s_start > span.last_row {
                break;
            }
            let lo = span.first_row.max(s_start);
            let hi = (span.last_row + 1).min(s_end);
            if lo >= hi {
                continue;
            }
            let row_start = (lo - s_start) as u32;
            let row_end = (hi - s_start) as u32;
            match &mut acc[i] {
                Some(entry) => {
                    entry.min_value = entry.min_value.min(span.min);
                    entry.max_value = entry.max_value.max(span.max);
                    entry.row_start = entry.row_start.min(row_start);
                    entry.row_end = entry.row_end.max(row_end);
                }
                slot @ None => {
                    *slot = Some(NumericLeafEntry {
                        min_value: span.min,
                        max_value: span.max,
                        shard_id: i as u32,
                        row_start,
                        row_end,
                    })
                }
            }
        }
    }

    let mut entries: Vec<NumericLeafEntry> = acc.into_iter().flatten().collect();
    // Value order, so `build_internal_pages`' split keys (each page's first
    // `min_value`) partition the tree the way a B+ tree's do.
    entries.sort_by(|a, b| {
        a.min_value
            .partial_cmp(&b.min_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries
}

/// Assemble a [`NumericIndex`] from row-ordered spans — the shared tail of
/// both builders.
fn numeric_index_from_spans(
    column_name: &str,
    spans: impl Iterator<Item = NumericSpan>,
    shard_row_ranges: &[(u64, u64)],
    fanout: u16,
) -> NumericIndex {
    let leaf_entries = numeric_leaves_from_spans(spans, shard_row_ranges);
    let leaf_pages: Vec<LeafPage> = leaf_entries
        .chunks(fanout as usize)
        .map(|chunk| LeafPage {
            entries: chunk.to_vec(),
        })
        .collect();
    let internal_pages = build_internal_pages(&leaf_pages, fanout);

    NumericIndex {
        column_name: column_name.to_string(),
        fanout,
        internal_pages,
        leaf_pages,
    }
}

/// Build serialized predicate-index bytes for a var RecordBatch. The
/// `shard_row_ranges` argument should be a single-shard range
/// `[(0, n_vars)]` — the engine treats var as a single shard for index
/// purposes (mirrors `scx-engine/tests/integration_tests.rs`).
pub fn build_var_predicate_index_bytes(
    var: &arrow::array::RecordBatch,
    shard_row_ranges: &[(u64, u64)],
    options: &PredicateIndexBuildOptions,
    outcomes: &mut Vec<BuildOutcome>,
    indexed_column_names: &mut Vec<String>,
) -> Result<Option<Vec<u8>>> {
    build_predicate_index_bytes_inner(
        var,
        shard_row_ranges,
        options,
        outcomes,
        indexed_column_names,
    )
}

/// CLI / Python conversion-time inputs for the predicate-index builder.
/// Mirrors the four `--index-*` CLI flags and the matching pyscx kwargs.
#[derive(Debug, Clone, Default)]
pub struct ConversionPredicateIndexOptions {
    /// Force-index these obs columns. Missing/unsupported columns
    /// produce a [`BuildOutcome::ForcedColumnError`] in the result.
    pub index_obs: Vec<String>,
    /// Force-index these var columns.
    pub index_var: Vec<String>,
    /// Named preset (`cellxgene` | `perturbseq` | `training`). Unknown
    /// names return an [`EngineError::UnknownIndexPreset`].
    pub index_preset: Option<String>,
    /// Cardinality cap for auto-detection when no forced/preset columns
    /// are supplied.
    pub index_auto_threshold: usize,
}

/// Result of [`build_and_write_conversion_predicate_indexes`]. Callers
/// walk the outcomes to demote preset skips to typed warnings and to
/// short-circuit on forced errors; `indexed_columns` is useful for
/// stamping provenance.
#[derive(Debug, Default)]
pub struct ConversionPredicateIndexResult {
    pub obs_outcomes: Vec<BuildOutcome>,
    pub obs_indexed_columns: Vec<String>,
    pub var_outcomes: Vec<BuildOutcome>,
    pub var_indexed_columns: Vec<String>,
}

/// Build both the obs and var predicate indexes for a conversion and
/// write them to `writer`. Returns the per-axis outcomes + indexed
/// column names; the caller maps outcomes to its own
/// warning/error types (see `scx-convert::pipeline` and
/// `pyscx::convert` for examples).
///
/// `obs_row_ranges` must reflect the actual on-disk CSR shard
/// boundaries. `var` is treated as a single shard `[(0, n_vars)]`
/// internally (matches `scx-engine/tests/integration_tests.rs`).
///
/// `high_cardinality_threshold` is fixed at 100_000 here — the cap
/// exists to keep a forced/preset column from blowing up the index;
/// auto-detect uses `options.index_auto_threshold` (default 1000).
pub fn build_and_write_conversion_predicate_indexes(
    writer: &mut scx_format_io::ScxWriter,
    obs: &arrow::array::RecordBatch,
    var: &arrow::array::RecordBatch,
    obs_row_ranges: &[(u64, u64)],
    n_vars: usize,
    options: &ConversionPredicateIndexOptions,
) -> Result<ConversionPredicateIndexResult> {
    // Delegate to the streaming variant with a one-shard iterator so
    // both entry points share a single implementation. Caller has the
    // assembled batch in hand — wrapping it in `std::iter::once` is
    // O(1).
    let obs_schema = obs.schema();
    build_and_write_conversion_predicate_indexes_streaming(
        writer,
        obs_schema,
        std::iter::once(Ok((obs.clone(), 0u64))),
        var,
        obs_row_ranges,
        n_vars,
        options,
    )
}

/// Streaming counterpart to [`build_and_write_conversion_predicate_indexes`].
///
/// Identical contract except `obs_shards` yields `(shard_batch,
/// shard_row_offset)` tuples one at a time — the obs predicate index
/// is built incrementally via [`ObsPredicateIndexBuilder`] so the
/// caller never has to assemble the full obs `RecordBatch` in memory.
/// Used by the sharded merge / append paths to keep peak RSS bounded
/// to one shard at a time.
///
/// `obs_schema` must describe the unified column layout of every
/// shard; the builder validates each shard's dtypes against it.
/// `var` is still treated as a single (small) batch — gene metadata
/// rarely overflows the 2 GB ceiling and the var predicate-index
/// path does not benefit meaningfully from shard streaming today.
pub fn build_and_write_conversion_predicate_indexes_streaming(
    writer: &mut scx_format_io::ScxWriter,
    obs_schema: arrow::datatypes::SchemaRef,
    obs_shards: impl IntoIterator<Item = Result<(arrow::array::RecordBatch, u64)>>,
    var: &arrow::array::RecordBatch,
    obs_row_ranges: &[(u64, u64)],
    n_vars: usize,
    options: &ConversionPredicateIndexOptions,
) -> Result<ConversionPredicateIndexResult> {
    streaming_impl(
        writer,
        obs_schema,
        obs_shards,
        var,
        obs_row_ranges,
        n_vars,
        options,
    )
}

fn streaming_impl(
    writer: &mut scx_format_io::ScxWriter,
    obs_schema: arrow::datatypes::SchemaRef,
    obs_shards: impl IntoIterator<Item = Result<(arrow::array::RecordBatch, u64)>>,
    var: &arrow::array::RecordBatch,
    obs_row_ranges: &[(u64, u64)],
    n_vars: usize,
    options: &ConversionPredicateIndexOptions,
) -> Result<ConversionPredicateIndexResult> {
    const HIGH_CARDINALITY_THRESHOLD: usize = 100_000;

    // Resolve preset up front so an unknown name fails before any
    // writer state changes.
    let (preset_obs, preset_var) = match options.index_preset.as_deref() {
        Some(name) => {
            let preset = index_preset_columns(name)
                .ok_or_else(|| EngineError::UnknownIndexPreset(name.to_string()))?;
            (
                preset
                    .obs_columns
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect::<Vec<_>>(),
                preset
                    .var_columns
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect::<Vec<_>>(),
            )
        }
        None => (Vec::new(), Vec::new()),
    };

    let mut result = ConversionPredicateIndexResult::default();

    // obs — stream shards through the builder.
    let obs_build_opts = PredicateIndexBuildOptions {
        forced_columns: options.index_obs.clone(),
        preset_columns: preset_obs,
        auto_threshold: options.index_auto_threshold,
        high_cardinality_threshold: HIGH_CARDINALITY_THRESHOLD,
    };
    let mut builder = ObsPredicateIndexBuilder::new(obs_schema, &obs_build_opts)?;
    for shard in obs_shards {
        let (batch, row_offset) = shard?;
        builder.push_shard_split(&batch, row_offset, obs_row_ranges)?;
    }
    let obs_bytes = builder.finish(
        obs_row_ranges,
        &mut result.obs_outcomes,
        &mut result.obs_indexed_columns,
    )?;
    if let Some(bytes) = obs_bytes {
        writer.write_obs_predicate_index(&bytes)?;
        // Populate per-shard catalog column stats from the index so query-time
        // `prune_shards_by_catalog_with_dict` can actually skip shards. Covers
        // convert (batch + streaming) and compact (eager + streaming) — both
        // route through here.
        apply_obs_shard_column_stats(writer, &bytes, obs_row_ranges.len())?;
    }

    // var (single shard) — gene metadata is small enough that the
    // batch-mode builder stays fine.
    let var_row_ranges: [(u64, u64); 1] = [(0, n_vars as u64)];
    let var_build_opts = PredicateIndexBuildOptions {
        forced_columns: options.index_var.clone(),
        preset_columns: preset_var,
        auto_threshold: options.index_auto_threshold,
        high_cardinality_threshold: HIGH_CARDINALITY_THRESHOLD,
    };
    let var_bytes = build_var_predicate_index_bytes(
        var,
        &var_row_ranges,
        &var_build_opts,
        &mut result.var_outcomes,
        &mut result.var_indexed_columns,
    )?;
    if let Some(bytes) = var_bytes {
        writer.write_var_predicate_index(&bytes)?;
    }

    Ok(result)
}

/// Derive per-shard catalog column statistics from a finished obs
/// [`PredicateIndex`].
///
/// `per_shard[k]` holds the stats for the k-th shard in the `shard_id` space
/// the index was built against (== catalog CSR-shard sorted order for
/// convert/merge/compact). For each categorical column we emit a
/// [`ColumnStat::CategoryBitset`] whose bit `i` is set iff dictionary value `i`
/// (the i-th entry, which is sorted ascending) is present in that shard — the
/// exact convention `pushdown::can_exclude_shard` checks
/// (`values.binary_search` → bit `i`). For each numeric column we fold the B+
/// tree leaf entries into a per-shard [`ColumnStat::MinMax`].
///
/// Precondition: every `shard_id` recorded in the index lies in `0..n_shards`
/// (the index is built against exactly `n_shards` `shard_row_ranges`). An
/// out-of-range id signals an index/catalog shard-space misalignment, which
/// would silently mis-place "value present" bits and risk incorrect shard
/// skipping; it trips a `debug_assert` and is otherwise dropped defensively.
///
/// **This positional `shard_id` → CSR-shard mapping is a write-time concern,
/// and multimodal safety does not rest on it alone.** Each modality's CSR shards
/// independently tile `[0, n_obs)`, so a flattened walk over "the i-th CSR
/// shard" stops meaning "shard_id i" the moment a second modality exists. Three
/// *independent* protections exist; none of them is "the file cannot contain an
/// index":
///
/// - **Write side (here).**
///   [`scx_format_io::writer::assign_csr_shard_column_stats`] counts only
///   `modality_id == 0` CSR entries and returns `ColumnStatsShardCountMismatch`
///   rather than mis-assigning, so these derived stats cannot land on a
///   multimodal file's shards
///   (`writer_tests::bulk_csr_shard_column_stats_refuses_a_multimodal_file`).
/// - **Level-2 read.** `collect::build_plan` forces `obs_predicate_index` to
///   `None` for every `modality_id != 0` pipeline, so `csr_shard_ranges_table`
///   never runs on a modality-scoped query even if an index section is present.
/// - **Level-1 read.** `pushdown::prune_shards_by_catalog_with_dict` reads the
///   `column_stats` already attached to each catalog entry and never consults an
///   index `shard_id`, so it is unaffected by this mapping either way.
///
/// **Index omission is a convention, not an invariant.** `scx convert` on an
/// h5mu emits `ConvertWarning::PredicateIndexSkippedMultimodal`,
/// `scx-ops::merge` records `multimodal_skip`, and `merge_multimodal` never
/// calls this — but `ScxWriter::write_obs_predicate_index` is public and accepts
/// a writer with registered modalities, so a file *can* carry one. That is why
/// the two read-side guards above matter and must not be removed on the grounds
/// that such a file "cannot exist".
///
/// Shipping multimodal indexing means giving `ShardRange` a modality scope, not
/// relaxing any of this.
pub fn derive_shard_column_stats(
    index: &PredicateIndex,
    n_shards: usize,
) -> Vec<Vec<scx_format_io::catalog::ColumnStat>> {
    use scx_format_io::catalog::ColumnStat;
    let mut per_shard: Vec<Vec<ColumnStat>> = vec![Vec::new(); n_shards];
    for column in &index.columns {
        match column {
            IndexedColumn::Categorical(cat) => {
                let hash = scx_format_io::column_name_hash(&cat.column_name);
                let n_values = cat.entries.len();
                let n_bytes = n_values.div_ceil(8);
                let mut bitsets: Vec<Vec<u8>> = vec![vec![0u8; n_bytes]; n_shards];
                for (bit, entry) in cat.entries.iter().enumerate() {
                    for range in &entry.shard_ranges {
                        let shard = range.shard_id as usize;
                        debug_assert!(
                            shard < n_shards,
                            "categorical shard_id {shard} >= n_shards {n_shards} — \
                             index/catalog shard-space misalignment"
                        );
                        if shard < n_shards {
                            bitsets[shard][bit / 8] |= 1 << (bit % 8);
                        }
                    }
                }
                for (shard, bitset) in bitsets.into_iter().enumerate() {
                    per_shard[shard].push(ColumnStat::CategoryBitset {
                        column_name_hash: hash,
                        bitset,
                    });
                }
            }
            IndexedColumn::Numeric(num) => {
                let hash = scx_format_io::column_name_hash(&num.column_name);
                let mut minmax: Vec<Option<(f64, f64)>> = vec![None; n_shards];
                for page in &num.leaf_pages {
                    for entry in &page.entries {
                        let shard = entry.shard_id as usize;
                        debug_assert!(
                            shard < n_shards,
                            "numeric shard_id {shard} >= n_shards {n_shards} — \
                             index/catalog shard-space misalignment"
                        );
                        if shard < n_shards {
                            let slot =
                                minmax[shard].get_or_insert((entry.min_value, entry.max_value));
                            slot.0 = slot.0.min(entry.min_value);
                            slot.1 = slot.1.max(entry.max_value);
                        }
                    }
                }
                for (shard, mm) in minmax.into_iter().enumerate() {
                    if let Some((min, max)) = mm {
                        per_shard[shard].push(ColumnStat::MinMax {
                            column_name_hash: hash,
                            min,
                            max,
                        });
                    }
                }
            }
        }
    }
    per_shard
}

/// Parse a serialized obs predicate index and attach the per-shard column stats
/// it implies to the writer's CSR shard catalog entries (one bulk pass).
///
/// `n_shards` must equal the number of CSR shards written and the length of the
/// `shard_row_ranges` the index was built with. Called after
/// `write_obs_predicate_index` in every fresh-writer build path.
pub fn apply_obs_shard_column_stats(
    writer: &mut scx_format_io::ScxWriter,
    obs_index_bytes: &[u8],
    n_shards: usize,
) -> Result<()> {
    let index = PredicateIndex::read_from(&mut std::io::Cursor::new(obs_index_bytes))?;
    let per_shard = derive_shard_column_stats(&index, n_shards);
    writer.set_csr_shard_column_stats_bulk(per_shard)?;
    Ok(())
}

/// Build a categorical index for a column.
pub fn build_categorical_index(
    column: &ArrayRef,
    column_name: &str,
    shard_row_ranges: &[(u64, u64)],
) -> CategoricalIndex {
    // Collect (value, global_row_idx) pairs
    let mut value_rows: BTreeMap<String, Vec<u64>> = BTreeMap::new();

    let n_rows = column.len();
    for row in 0..n_rows {
        if column.is_null(row) {
            continue;
        }
        let val = extract_string_value(column, row);
        if let Some(v) = val {
            value_rows.entry(v).or_default().push(row as u64);
        }
    }

    // Map global rows to shard ranges
    let mut entries: Vec<CategoricalEntry> = Vec::new();
    for (value, rows) in &value_rows {
        let shard_ranges = rows_to_shard_ranges(rows, shard_row_ranges);
        entries.push(CategoricalEntry {
            value: value.clone(),
            shard_ranges,
        });
    }
    // entries are already sorted lexicographically because BTreeMap is sorted

    CategoricalIndex {
        column_name: column_name.to_string(),
        entries,
    }
}

/// Build a numeric B+ tree index for an in-memory column.
///
/// Emits one leaf entry per shard, with that shard's exact `[min, max]` —
/// see [`numeric_leaves_from_spans`] for why per-shard is the right
/// granularity and why this path stays exact where the streaming one
/// summarises.
pub fn build_numeric_index(
    column: &ArrayRef,
    column_name: &str,
    shard_row_ranges: &[(u64, u64)],
    fanout: u16,
) -> NumericIndex {
    // One span per valued row. A one-row span can never straddle a shard
    // boundary, so this path stays exactly as precise as a per-row index
    // while emitting one leaf entry per shard — no accumulator, no sort.
    let spans = (0..column.len()).filter_map(|row| {
        if column.is_null(row) {
            return None;
        }
        extract_numeric_value(column, row).map(|v| NumericSpan {
            min: v,
            max: v,
            first_row: row as u64,
            last_row: row as u64,
        })
    });

    numeric_index_from_spans(column_name, spans, shard_row_ranges, fanout)
}

/// Build B+ tree internal pages from leaf pages, bottom-up.
fn build_internal_pages(leaf_pages: &[LeafPage], fanout: u16) -> Vec<InternalPage> {
    if leaf_pages.len() <= 1 {
        return vec![];
    }

    let mut all_internal: Vec<InternalPage> = Vec::new();
    // The "child indices" at the leaf level are just 0..leaf_pages.len()
    let mut current_level_indices: Vec<u32> = (0..leaf_pages.len() as u32).collect();
    // Extract the first key from each leaf page for split keys
    let mut current_keys: Vec<f64> = leaf_pages
        .iter()
        .map(|lp| lp.entries.first().map_or(f64::NAN, |e| e.min_value))
        .collect();

    loop {
        if current_level_indices.len() <= 1 {
            break;
        }

        let mut next_level_indices: Vec<u32> = Vec::new();
        let mut next_keys: Vec<f64> = Vec::new();

        // Group children into pages of `fanout+1` children each (fanout keys)
        let max_children = (fanout as usize) + 1;
        let mut i = 0;
        while i < current_level_indices.len() {
            let end = (i + max_children).min(current_level_indices.len());
            let children_slice = &current_level_indices[i..end];
            // Keys are the split values between children (skip the first in each group)
            let keys_slice: Vec<f64> = current_keys[i + 1..end].to_vec();
            let n_keys = keys_slice.len() as u16;

            let page_idx = all_internal.len() as u32;
            all_internal.push(InternalPage {
                n_keys,
                keys: keys_slice,
                children: children_slice.to_vec(),
            });

            next_level_indices.push(page_idx);
            next_keys.push(current_keys[i]); // first key of this group

            i = end;
        }

        if next_level_indices.len() <= 1 {
            break;
        }

        current_level_indices = next_level_indices;
        current_keys = next_keys;
    }

    all_internal
}

// ============================================================================
// Helper functions
// ============================================================================

/// Estimate the number of unique values in an Arrow array.
fn estimate_unique_values(col: &ArrayRef) -> usize {
    let dt = col.data_type();
    match dt {
        DataType::Dictionary(_, _) => {
            // Dictionary-encoded: unique count = dictionary length
            // For DictionaryArray<Int8/Int16/Int32>, get the values array length
            if let Some(dict) = col.as_any_dictionary_opt() {
                dict.values().len()
            } else {
                col.len() // fallback
            }
        }
        DataType::Utf8 => {
            let arr = col.as_string::<i32>();
            let mut seen = HashSet::new();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    seen.insert(arr.value(i).to_string());
                }
            }
            seen.len()
        }
        DataType::LargeUtf8 => {
            let arr = col.as_string::<i64>();
            let mut seen = HashSet::new();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    seen.insert(arr.value(i).to_string());
                }
            }
            seen.len()
        }
        _ if is_numeric_type(dt) => {
            // For numeric columns, sample-estimate uniqueness
            // Use a hash set on the first min(10000, len) values
            let n = col.len().min(10_000);
            let mut seen = HashSet::new();
            for i in 0..n {
                if !col.is_null(i) {
                    if let Some(v) = extract_numeric_value(col, i) {
                        seen.insert(v.to_bits());
                    }
                }
            }
            // Extrapolate if sampled
            if n < col.len() {
                let ratio = col.len() as f64 / n as f64;
                (seen.len() as f64 * ratio.sqrt()) as usize // rough Chao1-like estimate
            } else {
                seen.len()
            }
        }
        _ => col.len(), // unknown type, assume high cardinality
    }
}

/// True when two Arrow dtypes are interchangeable for predicate-index
/// purposes: both categorical (`Utf8` / `LargeUtf8`, or a dictionary over
/// one of those) or both numeric (a numeric type, or a dictionary over
/// one). Classification goes through [`logical_type`], so a
/// `Dictionary(_, V)` shard and a plain `V` shard are interchangeable —
/// which matters because `append` writes some columns plain where
/// `from_anndata` writes them dictionary-encoded. Used by
/// [`ObsPredicateIndexBuilder::push_shard`] to accept shards whose
/// per-shard upcast widens columns to `LargeUtf8` while the builder
/// was initialised with the input file's narrow `Utf8` schema.
fn column_class_compatible(a: &DataType, b: &DataType) -> bool {
    (is_categorical_type(a) && is_categorical_type(b))
        || (is_numeric_type(a) && is_numeric_type(b))
        || a == b
}

/// The type a column *logically* holds: a dictionary's value type, or the
/// type itself. Dictionary encoding is a storage detail — pandas writes
/// every `Categorical` that way regardless of what the categories are — so
/// classification must look through it. Both [`is_categorical_type`] and
/// [`is_numeric_type`] go through this, which is what keeps
/// `Dictionary(_, Int64)` and plain `Int64` from drifting into different
/// classes (see [`column_class_compatible`]).
fn logical_type(dt: &DataType) -> &DataType {
    match dt {
        DataType::Dictionary(_, value_type) => value_type.as_ref(),
        other => other,
    }
}

/// Check if a data type is categorical: a string, or a dictionary **whose
/// values are strings**.
///
/// The value type matters. A `Dictionary(_, Int64)` — what
/// `pd.Categorical([1, 2, 3])` becomes — is not categorical for index
/// purposes, because `build_categorical_index` can only extract string
/// values from a dictionary. Accepting it wrote an index with zero entries
/// and no outcome, leaving the column unreachable from the query engine.
/// Such a column routes to the numeric index instead;
/// a non-string, non-numeric value type (e.g. `Boolean`) is reported as an
/// unsupported dtype, exactly as the equivalent plain column already is.
fn is_categorical_type(dt: &DataType) -> bool {
    matches!(logical_type(dt), DataType::Utf8 | DataType::LargeUtf8)
}

/// Check if a data type is numeric, looking through dictionary encoding
/// so an integer- or float-valued pandas `Categorical` is indexed as the
/// numbers it holds.
fn is_numeric_type(dt: &DataType) -> bool {
    matches!(
        logical_type(dt),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Extract a string value from an Arrow array at the given row.
fn extract_string_value(col: &ArrayRef, row: usize) -> Option<String> {
    let dt = col.data_type();
    match dt {
        DataType::Utf8 => {
            let arr = col.as_string::<i32>();
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row).to_string())
            }
        }
        DataType::LargeUtf8 => {
            let arr = col.as_string::<i64>();
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row).to_string())
            }
        }
        DataType::Dictionary(_, _) => {
            let dict = col.as_any_dictionary_opt()?;
            let key = dictionary_key_at(dict, row)?;
            let values = dict.values();
            match values.data_type() {
                DataType::Utf8 => values
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .filter(|a| !a.is_null(key))
                    .map(|a| a.value(key).to_string()),
                DataType::LargeUtf8 => values
                    .as_any()
                    .downcast_ref::<arrow::array::LargeStringArray>()
                    .filter(|a| !a.is_null(key))
                    .map(|a| a.value(key).to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Resolve the dictionary key at `row` to an index into `values()`, for any
/// Arrow key width. `None` for a null key, a negative one, or one past the
/// end of `values()` (a malformed file must not panic on the values index —
/// see docs/conventions.md).
///
/// Shared by [`extract_string_value`] and [`extract_numeric_value`] so the
/// two cannot disagree about which key widths decode. The string path used
/// to hand-roll `Int32`/`Int8`/`Int16` only and silently returned `None` —
/// an *empty index* — for the rest; `widen_dictionary_keys` normalises
/// on-disk keys to `Int32`, but the builders also run on caller-supplied
/// batches, where `Int64` and unsigned keys occur.
///
/// **This must stay O(1).** It is called once per non-null row by both the
/// batch and the streaming index builders, so anything that touches the
/// whole key column here is quadratic in `n_obs`. `AnyDictionaryArray`
/// offers `normalized_keys()`, which looks like the tidy way to cover every
/// key width in one line and is not: it allocates and fills a
/// `Vec<usize>` over the entire column on **every call**, so an index build
/// that was linear became quadratic (measured end to end at 16k / 32k / 64k
/// rows: 0.080 s / 0.269 s / 1.067 s — ~4x per 2x rows). Dispatch on the key
/// type instead; a `downcast_ref` is a type-id check, not a scan.
fn dictionary_key_at(dict: &dyn arrow::array::AnyDictionaryArray, row: usize) -> Option<usize> {
    use arrow::array::PrimitiveArray;
    use arrow::datatypes::{
        Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    };

    let keys = dict.keys();
    if keys.is_null(row) {
        return None;
    }
    macro_rules! key_as_i128 {
        ($t:ty) => {
            keys.as_any()
                .downcast_ref::<PrimitiveArray<$t>>()
                .map(|k| k.value(row) as i128)
        };
    }
    // Widen through i128 so every key width — including UInt64 — is compared
    // in a domain that can hold it, and a negative key fails the conversion
    // below rather than wrapping to a huge index.
    let key = match keys.data_type() {
        DataType::Int8 => key_as_i128!(Int8Type),
        DataType::Int16 => key_as_i128!(Int16Type),
        DataType::Int32 => key_as_i128!(Int32Type),
        DataType::Int64 => key_as_i128!(Int64Type),
        DataType::UInt8 => key_as_i128!(UInt8Type),
        DataType::UInt16 => key_as_i128!(UInt16Type),
        DataType::UInt32 => key_as_i128!(UInt32Type),
        DataType::UInt64 => key_as_i128!(UInt64Type),
        _ => None,
    }?;
    let key = usize::try_from(key).ok()?;
    // Bounds check is belt-and-braces: `DictionaryArray::try_new` rejects an
    // out-of-range key at construction, so this can only fire on an array
    // that reached memory another way. It costs one comparison and turns a
    // would-be panic on the values index into a skipped row.
    (key < dict.values().len()).then_some(key)
}

/// Extract a numeric value from an Arrow array as f64.
fn extract_numeric_value(col: &ArrayRef, row: usize) -> Option<f64> {
    if col.is_null(row) {
        return None;
    }
    match col.data_type() {
        DataType::Int8 => col
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|a| a.value(row) as f64),
        DataType::Int16 => col
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|a| a.value(row) as f64),
        DataType::Int32 => col
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| a.value(row) as f64),
        DataType::Int64 => col
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| a.value(row) as f64),
        DataType::UInt8 => col
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|a| a.value(row) as f64),
        DataType::UInt16 => col
            .as_any()
            .downcast_ref::<UInt16Array>()
            .map(|a| a.value(row) as f64),
        DataType::UInt32 => col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|a| a.value(row) as f64),
        DataType::UInt64 => col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|a| a.value(row) as f64),
        DataType::Float32 => col
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| a.value(row) as f64),
        DataType::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| a.value(row)),
        // A numeric-valued pandas `Categorical` (`pd.Categorical([1, 2, 3])`
        // → `Dictionary(_, Int64)`). Resolve the key and recurse into the
        // values array, which is one of the arms above.
        DataType::Dictionary(_, _) => {
            let dict = col.as_any_dictionary_opt()?;
            let key = dictionary_key_at(dict, row)?;
            extract_numeric_value(dict.values(), key)
        }
        _ => None,
    }
}

/// Map global rows to ShardRanges (local indices within shards).
fn rows_to_shard_ranges(rows: &[u64], shard_row_ranges: &[(u64, u64)]) -> Vec<ShardRange> {
    // Group rows by shard, tracking contiguous local ranges
    let mut ranges: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();

    for &global_row in rows {
        let Some((shard_id, local_row)) = global_row_to_shard(global_row, shard_row_ranges) else {
            continue; // row doesn't belong to any shard — skip
        };
        let shard_ranges = ranges.entry(shard_id).or_default();
        // Try to extend the last range
        if let Some(last) = shard_ranges.last_mut() {
            if local_row == last.1 {
                last.1 = local_row + 1;
                continue;
            }
        }
        shard_ranges.push((local_row, local_row + 1));
    }

    let mut result = Vec::new();
    for (shard_id, row_ranges) in &ranges {
        for &(start, end) in row_ranges {
            result.push(ShardRange {
                shard_id: *shard_id,
                row_start: start,
                row_end: end,
            });
        }
    }
    result
}

/// Find which shard a global row belongs to, returning (shard_id, local_row).
/// Returns `None` if the global row falls in a gap between shards or is out of range,
/// rather than silently computing a garbage local_row via underflow.
fn global_row_to_shard(global_row: u64, shard_row_ranges: &[(u64, u64)]) -> Option<(u32, u32)> {
    for (i, &(start, end)) in shard_row_ranges.iter().enumerate() {
        if global_row >= start && global_row < end {
            return Some((i as u32, (global_row - start) as u32));
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "index_tests.rs"]
mod tests;
