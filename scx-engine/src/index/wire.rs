//! Predicate index binary layout — docs/format.md (Predicate Indexes).
//!
//! Both on-disk versions live here. v1 uses narrow (u16/u32) counters; v2
//! widens them, and [`requires_v2_encoding`] decides which a given index needs.
//! The version byte on disk is the source of truth on read.

use std::io::{Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{EngineError, Result};

use super::{
    CategoricalEntry, CategoricalIndex, IndexedColumn, InternalPage, LeafPage, NumericIndex,
    NumericLeafEntry, PredicateIndex, ShardRange,
};

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
pub(crate) fn requires_v2_encoding(index: &PredicateIndex) -> bool {
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

impl PredicateIndex {
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
