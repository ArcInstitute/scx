// Predicate index read/write — docs/format.md (Predicate Indexes)
//
// Implements sorted value-to-shard-range mappings that enable fine-grained
// row-level filtering within shards. This is level 2 of the two-level pushdown
// strategy (docs/api.md (Query engine, optimizations)).

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

/// Numeric index using a B+ tree for range queries.
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

/// A leaf entry mapping a numeric value range to a shard row range.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericLeafEntry {
    pub min_value: f64,
    pub max_value: f64,
    pub shard_id: u32,
    pub row_start: u32,
    pub row_end: u32,
}

/// Hash index for high-cardinality columns (>10K unique values).
/// Stretch goal — not implemented in initial Phase 2.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct HashIndex {
    pub column_name: String,
    pub mph_data: Vec<u8>,
    pub entries: Vec<CategoricalEntry>,
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

impl PredicateIndex {
    /// Serialize the predicate index per docs/format.md (Predicate Indexes).
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u8(self.version)?;
        w.write_u16::<LittleEndian>(self.columns.len() as u16)?;
        for col in &self.columns {
            match col {
                IndexedColumn::Categorical(cat) => {
                    // column name
                    let name_bytes = cat.column_name.as_bytes();
                    w.write_u16::<LittleEndian>(name_bytes.len() as u16)?;
                    w.write_all(name_bytes)?;
                    // column_type = 0 (categorical)
                    w.write_u8(0)?;
                    // n_entries
                    w.write_u32::<LittleEndian>(cat.entries.len() as u32)?;
                    for entry in &cat.entries {
                        let val_bytes = entry.value.as_bytes();
                        w.write_u16::<LittleEndian>(val_bytes.len() as u16)?;
                        w.write_all(val_bytes)?;
                        w.write_u16::<LittleEndian>(entry.shard_ranges.len() as u16)?;
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
                    w.write_u16::<LittleEndian>(name_bytes.len() as u16)?;
                    w.write_all(name_bytes)?;
                    // column_type = 1 (numeric)
                    w.write_u8(1)?;
                    // n_entries (total leaf entries for the header)
                    let total_entries: u32 = num
                        .leaf_pages
                        .iter()
                        .map(|lp| lp.entries.len() as u32)
                        .sum();
                    w.write_u32::<LittleEndian>(total_entries)?;
                    // B+ tree metadata
                    w.write_u16::<LittleEndian>(num.fanout)?;
                    w.write_u32::<LittleEndian>(num.leaf_pages.len() as u32)?;
                    w.write_u32::<LittleEndian>(num.internal_pages.len() as u32)?;
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
                        w.write_u32::<LittleEndian>(page.entries.len() as u32)?;
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
/// path) and `pyscx/src/anndata.rs` (Python path) so both surfaces
/// emit the same message. — E2-2026-05-20.
///
/// Lives in `scx-engine` rather than `scx-convert` because pyscx
/// depends on `scx-engine` unconditionally but only pulls in
/// `scx-convert` under the `hdf5` feature; the helper must remain
/// callable from `pyscx::anndata::build_and_write_predicate_indexes_inline`
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
/// When `show_all` is true and `available.len() <= 64`, lists every
/// column when strsim has a near-match the user
/// is "on the right page" and benefits from seeing the full list). The
/// 64-column cap keeps the worst-case message bounded for atlases with
/// hundreds of obs columns. Otherwise falls back to the historical
/// first-8 preview with `", ..."` truncation marker.
///
/// Empty `available` yields the empty-axis fallback that hints at no
/// obs/var metadata being present in the file.
fn render_available_columns(axis: &str, available: &[String], show_all: bool) -> String {
    if available.is_empty() {
        return format!("Available {axis} columns: [] (this h5ad has no {axis} metadata).");
    }
    let show_all = show_all && available.len() <= 64;
    if show_all {
        let items: Vec<&str> = available.iter().map(String::as_str).collect();
        return format!("Available {axis} columns: {items:?}.");
    }
    let preview_n = available.len().min(8);
    let preview: Vec<&str> = available[..preview_n].iter().map(String::as_str).collect();
    let suffix = if available.len() > preview_n {
        ", ..."
    } else {
        ""
    };
    format!("Available {axis} columns: {preview:?}{suffix}.")
}

/// Shared suffix builder for [`forced_column_missing_message`] and
/// [`column_not_found_message`]. Computes the strsim suggestion first
/// so the available-columns renderer can decide whether to show all
/// (suggestion present → user is on the right page)
/// or fall back to the 8-column preview.
fn column_suggestion_suffix(axis: &str, column: &str, available: &[String]) -> String {
    let suggestion = best_match(column, available);
    let mut msg = render_available_columns(axis, available, suggestion.is_some());
    if let Some(s) = suggestion {
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
    // If ANY missing column has a strsim near-match
    // in `available`, the user is "on the right page" — show the full
    // available-columns list (capped at 64) instead of the 8-column
    // preview so they can scan and pick the right name.
    let mut any_suggestion = false;
    for column in missing {
        let suggestion = best_match(column, available);
        if suggestion.is_some() {
            any_suggestion = true;
        }
        match suggestion {
            Some(s) => msg.push_str(&format!("\n  - '{column}': did you mean '{s}'?")),
            None => msg.push_str(&format!("\n  - '{column}'")),
        }
    }
    msg.push('\n');
    msg.push_str(&render_available_columns(axis, available, any_suggestion));
    msg
}

#[cfg(test)]
mod forced_column_missing_message_tests {
    use super::forced_column_missing_message;

    #[test]
    fn lists_available_columns() {
        let avail = vec![
            "total_counts".to_string(),
            "n_genes_by_counts".to_string(),
            "pct_counts_mt".to_string(),
        ];
        let msg = forced_column_missing_message("obs", "nonexistent_column", &avail);
        assert!(
            msg.contains("forced obs index column 'nonexistent_column'"),
            "{msg}"
        );
        assert!(msg.contains("missing column."), "{msg}");
        assert!(msg.contains("total_counts"), "{msg}");
        assert!(msg.contains("n_genes_by_counts"), "{msg}");
        assert!(!msg.contains("Did you mean"), "{msg}");
    }

    #[test]
    fn suggests_typo() {
        let avail = vec!["total_counts".to_string(), "n_genes_by_counts".to_string()];
        let msg = forced_column_missing_message("obs", "totl_counts", &avail);
        assert!(msg.contains("Did you mean 'total_counts'?"), "{msg}");
    }

    #[test]
    fn no_suggestion_when_far() {
        let avail = vec!["foo".to_string(), "bar".to_string()];
        let msg = forced_column_missing_message("obs", "cell_type", &avail);
        assert!(!msg.contains("Did you mean"), "{msg}");
        assert!(msg.contains("foo") && msg.contains("bar"), "{msg}");
    }

    #[test]
    fn handles_empty_available() {
        let msg = forced_column_missing_message("obs", "total_counts", &[]);
        assert!(
            msg.contains("Available obs columns: [] (this h5ad has no obs metadata)"),
            "{msg}"
        );
        assert!(!msg.contains("Did you mean"), "{msg}");
    }

    #[test]
    fn truncates_long_available_lists() {
        let avail: Vec<String> = (0..20).map(|i| format!("col_{i}")).collect();
        let msg = forced_column_missing_message("var", "missing", &avail);
        assert!(msg.contains(", ..."), "should signal truncation: {msg}");
        assert!(msg.contains("col_0") && msg.contains("col_7"), "{msg}");
        assert!(
            !msg.contains("col_15"),
            "should not list past index 7: {msg}"
        );
    }

    // E1-2026-05-20-Tier2: when the typo has a near-match in available,
    // the user is on the right page — show ALL columns (up to the 64-col
    // cap) instead of truncating to 8 so they can scan past index 7.
    #[test]
    fn shows_all_columns_when_strsim_suggestion_present() {
        // 28 obs columns mirroring the census_500k.scx layout; `raw_sum`
        // is at index 11 (past the 8-column cutoff). The strsim match
        // for `raw_summ` should be `raw_sum`, which triggers show-all.
        let mut avail: Vec<String> = (0..28).map(|i| format!("col_{i:02}")).collect();
        avail[11] = "raw_sum".to_string();
        let msg = forced_column_missing_message("obs", "raw_summ", &avail);
        assert!(
            msg.contains("Did you mean 'raw_sum'?"),
            "suggestion gate: {msg}"
        );
        assert!(
            !msg.contains(", ..."),
            "show-all path should NOT emit the truncation marker: {msg}"
        );
        assert!(msg.contains("col_00"), "should show first column: {msg}");
        assert!(
            msg.contains("col_27"),
            "should show LAST column (past 8-col preview): {msg}"
        );
        assert!(msg.contains("raw_sum"), "{msg}");
    }

    // E1 corner: empty-suggestion case keeps the 8-column preview to
    // avoid overwhelming a "fishing" user with hundreds of column names.
    #[test]
    fn keeps_preview_when_no_strsim_suggestion() {
        let avail: Vec<String> = (0..28).map(|i| format!("col_{i:02}")).collect();
        let msg = forced_column_missing_message("obs", "totally_unrelated", &avail);
        assert!(
            !msg.contains("Did you mean"),
            "no suggestion expected: {msg}"
        );
        assert!(msg.contains(", ..."), "should keep truncation: {msg}");
        assert!(
            !msg.contains("col_27"),
            "should NOT show past index 7: {msg}"
        );
    }

    // The 64-column cap guards the worst-case payload size: an atlas
    // with > 64 obs columns falls back to the 8-column preview even
    // when a suggestion is present.
    #[test]
    fn cap_at_64_columns_falls_back_to_preview() {
        let mut avail: Vec<String> = (0..70).map(|i| format!("col_{i:03}")).collect();
        avail[50] = "raw_sum".to_string();
        let msg = forced_column_missing_message("obs", "raw_summ", &avail);
        assert!(msg.contains("Did you mean 'raw_sum'?"), "{msg}");
        assert!(
            msg.contains(", ..."),
            "should truncate past the 64-col cap: {msg}"
        );
        assert!(
            !msg.contains("col_050"),
            "the suggested column lives at index 50 — past the 8-col preview: {msg}"
        );
    }
}

#[cfg(test)]
mod forced_columns_missing_message_tests {
    use super::forced_columns_missing_message;

    fn census_obs_columns() -> Vec<String> {
        vec![
            "soma_joinid".to_string(),
            "dataset_id".to_string(),
            "cell_type".to_string(),
            "raw_sum".to_string(),
            "tissue".to_string(),
            "disease".to_string(),
        ]
    }

    #[test]
    fn aggregates_multiple_misses_with_suggestions() {
        // F6-2026-05-20-Tier2: both `raw_summ` and `cell_typ` should be
        // surfaced in a single error, each with its own strsim hint.
        let missing = vec!["raw_summ".to_string(), "cell_typ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &census_obs_columns());
        assert!(
            msg.contains("2 forced obs index columns are missing"),
            "should aggregate count + axis: {msg}"
        );
        assert!(
            msg.contains("- 'raw_summ': did you mean 'raw_sum'?"),
            "should include first typo + suggestion: {msg}"
        );
        assert!(
            msg.contains("- 'cell_typ': did you mean 'cell_type'?"),
            "should include second typo + suggestion: {msg}"
        );
        assert!(
            msg.contains("Available obs columns"),
            "should include available-columns footer: {msg}"
        );
    }

    #[test]
    fn aggregates_misses_without_near_suggestion() {
        let missing = vec![
            "totally_unrelated".to_string(),
            "another_unrelated".to_string(),
        ];
        let msg = forced_columns_missing_message("obs", &missing, &census_obs_columns());
        assert!(
            msg.contains("2 forced obs index columns are missing"),
            "{msg}"
        );
        // Both names without suggestions should appear on their own bullets.
        assert!(msg.contains("- 'totally_unrelated'"), "{msg}");
        assert!(msg.contains("- 'another_unrelated'"), "{msg}");
        assert!(
            !msg.contains("Did you mean") && !msg.contains("did you mean"),
            "no near match in either case: {msg}"
        );
    }

    #[test]
    fn single_miss_delegates_to_singular_helper() {
        // The singular path keeps the historical wording so we don't
        // gratuitously change byte-for-byte output of the single-miss
        // surface (the most common one in practice).
        let missing = vec!["raw_summ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &census_obs_columns());
        assert!(
            msg.contains("forced obs index column 'raw_summ': missing column."),
            "should use the singular wording: {msg}"
        );
        assert!(msg.contains("Did you mean 'raw_sum'?"), "{msg}");
        // The aggregate header MUST NOT appear when N == 1.
        assert!(
            !msg.contains("forced obs index columns are missing"),
            "should not emit the aggregate header for single miss: {msg}"
        );
    }

    #[test]
    fn empty_available_columns_is_handled() {
        let missing = vec!["raw_summ".to_string(), "cell_typ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &[]);
        assert!(
            msg.contains("2 forced obs index columns are missing"),
            "{msg}"
        );
        assert!(
            msg.contains("[] (this h5ad has no obs metadata)"),
            "should render the empty-axis footer: {msg}"
        );
    }

    // E1-2026-05-20-Tier2: aggregate path inherits the show-all-when-
    // strsim-matches behaviour from `render_available_columns`. If ANY
    // of the misses has a near-match, the full column list (≤ 64) shows.
    #[test]
    fn aggregate_shows_all_columns_when_any_miss_has_suggestion() {
        let mut avail: Vec<String> = (0..28).map(|i| format!("col_{i:02}")).collect();
        avail[11] = "raw_sum".to_string();
        avail[14] = "cell_type".to_string();
        let missing = vec!["raw_summ".to_string(), "cell_typ".to_string()];
        let msg = forced_columns_missing_message("obs", &missing, &avail);
        assert!(
            msg.contains("did you mean 'raw_sum'?") && msg.contains("did you mean 'cell_type'?"),
            "both suggestions should render: {msg}"
        );
        assert!(
            !msg.contains(", ..."),
            "show-all path should NOT truncate: {msg}"
        );
        assert!(
            msg.contains("col_27"),
            "should show LAST column (past 8-col preview): {msg}"
        );
    }
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
/// `pyscx::anndata` for examples).
///
/// `obs_row_ranges` must reflect the actual on-disk CSR shard
/// boundaries. `var` is treated as a single shard `[(0, n_vars)]`
/// internally (matches `scx-engine/tests/integration_tests.rs`).
///
/// `high_cardinality_threshold` is fixed at 100_000 here — the cap
/// exists to keep a forced/preset column from blowing up the index;
/// auto-detect uses `options.index_auto_threshold` (default 1000).
pub fn build_and_write_conversion_predicate_indexes(
    writer: &mut scx_format::ScxWriter,
    obs: &arrow::array::RecordBatch,
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

    // obs
    let obs_build_opts = PredicateIndexBuildOptions {
        forced_columns: options.index_obs.clone(),
        preset_columns: preset_obs,
        auto_threshold: options.index_auto_threshold,
        high_cardinality_threshold: HIGH_CARDINALITY_THRESHOLD,
    };
    let obs_bytes = build_obs_predicate_index_bytes(
        obs,
        obs_row_ranges,
        &obs_build_opts,
        &mut result.obs_outcomes,
        &mut result.obs_indexed_columns,
    )?;
    if let Some(bytes) = obs_bytes {
        writer.write_obs_predicate_index(&bytes)?;
    }

    // var (single shard)
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

/// Build a numeric B+ tree index for a column.
pub fn build_numeric_index(
    column: &ArrayRef,
    column_name: &str,
    shard_row_ranges: &[(u64, u64)],
    fanout: u16,
) -> NumericIndex {
    // Collect (value, global_row_idx) pairs
    let mut value_rows: Vec<(f64, u64)> = Vec::new();
    let n_rows = column.len();
    for row in 0..n_rows {
        if column.is_null(row) {
            continue;
        }
        if let Some(v) = extract_numeric_value(column, row) {
            value_rows.push((v, row as u64));
        }
    }

    // Sort by value
    value_rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if value_rows.is_empty() {
        return NumericIndex {
            column_name: column_name.to_string(),
            fanout,
            internal_pages: vec![],
            leaf_pages: vec![],
        };
    }

    // Group consecutive values into leaf entries, one per shard range
    // Each leaf entry covers a contiguous value range within one shard.
    let mut leaf_entries: Vec<NumericLeafEntry> = Vec::new();
    for &(val, global_row) in &value_rows {
        let Some((shard_id, local_row)) = global_row_to_shard(global_row, shard_row_ranges) else {
            continue; // row doesn't belong to any shard — skip
        };
        // Try to extend the last entry if same shard and adjacent row
        if let Some(last) = leaf_entries.last_mut() {
            if last.shard_id == shard_id && local_row == last.row_end {
                last.max_value = val;
                last.row_end = local_row + 1;
                continue;
            }
        }
        leaf_entries.push(NumericLeafEntry {
            min_value: val,
            max_value: val,
            shard_id,
            row_start: local_row,
            row_end: local_row + 1,
        });
    }

    // Partition leaf entries into leaf pages (up to `fanout` entries per page)
    let leaf_pages: Vec<LeafPage> = leaf_entries
        .chunks(fanout as usize)
        .map(|chunk| LeafPage {
            entries: chunk.to_vec(),
        })
        .collect();

    // Build internal pages bottom-up
    let internal_pages = build_internal_pages(&leaf_pages, fanout);

    NumericIndex {
        column_name: column_name.to_string(),
        fanout,
        internal_pages,
        leaf_pages,
    }
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

/// Check if a data type is categorical (string or dictionary).
fn is_categorical_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Dictionary(_, _)
    )
}

/// Check if a data type is numeric.
fn is_numeric_type(dt: &DataType) -> bool {
    matches!(
        dt,
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
            // Cast to StringArray view of dictionary values
            if let Some(dict) = col.as_any_dictionary_opt() {
                let keys = dict.keys();
                let values = dict.values();
                if keys.is_null(row) {
                    return None;
                }
                let key = keys
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .map(|k| k.value(row) as usize)
                    .or_else(|| {
                        keys.as_any()
                            .downcast_ref::<arrow::array::Int8Array>()
                            .map(|k| k.value(row) as usize)
                    })
                    .or_else(|| {
                        keys.as_any()
                            .downcast_ref::<arrow::array::Int16Array>()
                            .map(|k| k.value(row) as usize)
                    })?;
                values
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .map(|str_arr| str_arr.value(key).to_string())
            } else {
                None
            }
        }
        _ => None,
    }
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
mod tests {
    use super::*;
    use arrow::array::{Int32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::io::Cursor;
    use std::sync::Arc;

    // -------------------------------------------------------------------
    // C2 Tests: Serialization round-trips
    // -------------------------------------------------------------------

    #[test]
    fn roundtrip_empty_index() {
        let index = PredicateIndex {
            version: 1,
            columns: vec![],
        };
        let mut buf = Vec::new();
        index.write_to(&mut buf).unwrap();

        let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, index);
    }

    /// Regression: a 10-byte malformed input that declared
    /// `n_cat_entries = 0x2d000000` (~755 million) used to trigger a
    /// ~36 GB `Vec::with_capacity` and OOM the process. Found by the
    /// Phase 9 `fuzz_predicate_index` libfuzzer target on its first
    /// 10-second run. The reader now rejects the input with
    /// `InvalidData` via [`MAX_CATEGORICAL_ENTRIES`].
    #[test]
    fn read_from_rejects_oversized_categorical_count() {
        // version | n_columns(LE) | name_len(LE) | column_type | n_entries(LE)
        //   0xfb  |   0x000a      |   0x0000     |    0x00     |  0x2d000000
        let crash_input: [u8; 10] = [0xfb, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2d];
        let err = PredicateIndex::read_from(&mut Cursor::new(&crash_input[..]))
            .expect_err("oversized n_entries must be rejected, not allocated");
        let msg = format!("{err}");
        assert!(
            msg.contains("categorical entries") && msg.contains("exceeds maximum"),
            "expected 'categorical entries ... exceeds maximum' error, got: {msg}"
        );
    }

    #[test]
    fn roundtrip_categorical_index() {
        let index = PredicateIndex {
            version: 1,
            columns: vec![IndexedColumn::Categorical(CategoricalIndex {
                column_name: "cell_type".to_string(),
                entries: vec![
                    CategoricalEntry {
                        value: "B cell".to_string(),
                        shard_ranges: vec![
                            ShardRange {
                                shard_id: 0,
                                row_start: 10,
                                row_end: 20,
                            },
                            ShardRange {
                                shard_id: 2,
                                row_start: 0,
                                row_end: 15,
                            },
                        ],
                    },
                    CategoricalEntry {
                        value: "NK cell".to_string(),
                        shard_ranges: vec![
                            ShardRange {
                                shard_id: 1,
                                row_start: 5,
                                row_end: 30,
                            },
                            ShardRange {
                                shard_id: 3,
                                row_start: 0,
                                row_end: 50,
                            },
                            ShardRange {
                                shard_id: 4,
                                row_start: 10,
                                row_end: 40,
                            },
                        ],
                    },
                    CategoricalEntry {
                        value: "T cell".to_string(),
                        shard_ranges: vec![
                            ShardRange {
                                shard_id: 0,
                                row_start: 0,
                                row_end: 10,
                            },
                            ShardRange {
                                shard_id: 1,
                                row_start: 0,
                                row_end: 5,
                            },
                            ShardRange {
                                shard_id: 2,
                                row_start: 15,
                                row_end: 50,
                            },
                        ],
                    },
                ],
            })],
        };
        let mut buf = Vec::new();
        index.write_to(&mut buf).unwrap();

        let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, index);
    }

    #[test]
    fn roundtrip_numeric_index() {
        let index = PredicateIndex {
            version: 1,
            columns: vec![IndexedColumn::Numeric(NumericIndex {
                column_name: "n_genes".to_string(),
                fanout: 4,
                internal_pages: vec![InternalPage {
                    n_keys: 1,
                    keys: vec![500.0],
                    children: vec![0, 1],
                }],
                leaf_pages: vec![
                    LeafPage {
                        entries: vec![
                            NumericLeafEntry {
                                min_value: 100.0,
                                max_value: 200.0,
                                shard_id: 0,
                                row_start: 0,
                                row_end: 50,
                            },
                            NumericLeafEntry {
                                min_value: 200.0,
                                max_value: 500.0,
                                shard_id: 1,
                                row_start: 0,
                                row_end: 100,
                            },
                        ],
                    },
                    LeafPage {
                        entries: vec![
                            NumericLeafEntry {
                                min_value: 500.0,
                                max_value: 800.0,
                                shard_id: 2,
                                row_start: 0,
                                row_end: 80,
                            },
                            NumericLeafEntry {
                                min_value: 800.0,
                                max_value: 5000.0,
                                shard_id: 3,
                                row_start: 0,
                                row_end: 200,
                            },
                        ],
                    },
                ],
            })],
        };
        let mut buf = Vec::new();
        index.write_to(&mut buf).unwrap();

        let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, index);
    }

    #[test]
    fn roundtrip_mixed_index() {
        let index = PredicateIndex {
            version: 1,
            columns: vec![
                IndexedColumn::Categorical(CategoricalIndex {
                    column_name: "cell_type".to_string(),
                    entries: vec![CategoricalEntry {
                        value: "T cell".to_string(),
                        shard_ranges: vec![ShardRange {
                            shard_id: 0,
                            row_start: 0,
                            row_end: 10,
                        }],
                    }],
                }),
                IndexedColumn::Numeric(NumericIndex {
                    column_name: "n_genes".to_string(),
                    fanout: 4,
                    internal_pages: vec![],
                    leaf_pages: vec![LeafPage {
                        entries: vec![NumericLeafEntry {
                            min_value: 100.0,
                            max_value: 5000.0,
                            shard_id: 0,
                            row_start: 0,
                            row_end: 100,
                        }],
                    }],
                }),
            ],
        };
        let mut buf = Vec::new();
        index.write_to(&mut buf).unwrap();
        let decoded = PredicateIndex::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, index);
    }

    // -------------------------------------------------------------------
    // C3 Tests: Index construction
    // -------------------------------------------------------------------

    fn make_obs_batch() -> RecordBatch {
        // 12 rows, 2 shards: shard 0 = rows 0..6, shard 1 = rows 6..12
        let schema = Schema::new(vec![
            Field::new("cell_type", DataType::Utf8, false),
            Field::new("n_genes", DataType::Int32, false),
        ]);
        let cell_types = StringArray::from(vec![
            "T cell", "B cell", "T cell", "NK cell", "B cell", "T cell", "NK cell", "T cell",
            "B cell", "NK cell", "NK cell", "T cell",
        ]);
        let n_genes = Int32Array::from(vec![
            200, 300, 150, 450, 500, 250, 600, 100, 350, 700, 800, 400,
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(cell_types), Arc::new(n_genes)],
        )
        .unwrap()
    }

    #[test]
    fn build_categorical_index_correct_entries() {
        let batch = make_obs_batch();
        let col = batch.column(0);
        let shard_ranges = vec![(0u64, 6u64), (6, 12)];

        let index = build_categorical_index(col, "cell_type", &shard_ranges);
        assert_eq!(index.column_name, "cell_type");
        assert_eq!(index.entries.len(), 3); // B cell, NK cell, T cell (sorted)
        assert_eq!(index.entries[0].value, "B cell");
        assert_eq!(index.entries[1].value, "NK cell");
        assert_eq!(index.entries[2].value, "T cell");

        // B cell appears in rows 1, 4 (shard 0) and 8 (shard 1)
        let b_cell = &index.entries[0];
        assert!(b_cell.shard_ranges.iter().any(|r| r.shard_id == 0));
        assert!(b_cell.shard_ranges.iter().any(|r| r.shard_id == 1));
    }

    #[test]
    fn build_numeric_index_valid_btree() {
        let batch = make_obs_batch();
        let col = batch.column(1);
        let shard_ranges = vec![(0u64, 6u64), (6, 12)];

        let index = build_numeric_index(col, "n_genes", &shard_ranges, 4);
        assert_eq!(index.column_name, "n_genes");
        assert!(!index.leaf_pages.is_empty());

        // All leaf entries should have valid min <= max
        for page in &index.leaf_pages {
            for entry in &page.entries {
                assert!(entry.min_value <= entry.max_value);
            }
        }
    }

    #[test]
    fn build_indexes_auto_detect() {
        let batch = make_obs_batch();
        let shard_ranges = vec![(0u64, 6u64), (6, 12)];

        // Empty indexed_columns → auto-detect
        let index = build_indexes(&batch, &shard_ranges, &[]).unwrap();
        assert!(!index.columns.is_empty());
        // Both cell_type and n_genes should be indexed (both have <1000 unique)
        assert_eq!(index.columns.len(), 2);
    }

    #[test]
    fn build_indexes_high_cardinality_skipped() {
        // Create a column with >10K unique values
        let n = 11_000;
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        let ids = Int32Array::from((0..n).collect::<Vec<i32>>());
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids)]).unwrap();
        let shard_ranges = vec![(0u64, n as u64)];

        let index = build_indexes(&batch, &shard_ranges, &["id".to_string()]).unwrap();
        // Should be skipped because >10K unique
        assert!(index.columns.is_empty());
    }

    #[test]
    fn build_indexes_empty_batch() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let batch = RecordBatch::new_empty(Arc::new(schema));
        let shard_ranges: Vec<(u64, u64)> = vec![];

        let index = build_indexes(&batch, &shard_ranges, &[]).unwrap();
        // Empty batch auto-detect should produce empty or index with empty entries
        assert!(
            index.columns.is_empty()
                || index.columns.iter().all(|c| match c {
                    IndexedColumn::Categorical(cat) => cat.entries.is_empty(),
                    IndexedColumn::Numeric(num) => num.leaf_pages.is_empty(),
                })
        );
    }

    #[test]
    fn build_indexes_explicit_columns() {
        let batch = make_obs_batch();
        let shard_ranges = vec![(0u64, 6u64), (6, 12)];

        let index = build_indexes(&batch, &shard_ranges, &["cell_type".to_string()]).unwrap();
        assert_eq!(index.columns.len(), 1);
        match &index.columns[0] {
            IndexedColumn::Categorical(cat) => {
                assert_eq!(cat.column_name, "cell_type");
            }
            _ => panic!("expected categorical"),
        }
    }

    // --- Phase 5a ---

    #[test]
    fn index_preset_columns_known_names() {
        for name in ["cellxgene", "perturbseq", "training"] {
            let p = index_preset_columns(name).expect("known preset");
            assert!(!p.obs_columns.is_empty());
        }
        assert!(index_preset_columns("unknown").is_none());
    }

    #[test]
    fn build_obs_predicate_index_bytes_forced_missing_errors() {
        let batch = make_obs_batch();
        let shard_ranges = vec![(0u64, 12u64)];
        let opts = PredicateIndexBuildOptions {
            forced_columns: vec!["does_not_exist".to_string()],
            preset_columns: vec![],
            auto_threshold: 1000,
            high_cardinality_threshold: 100_000,
        };
        let mut outcomes = Vec::new();
        let mut names = Vec::new();
        let bytes = build_obs_predicate_index_bytes(
            &batch,
            &shard_ranges,
            &opts,
            &mut outcomes,
            &mut names,
        )
        .unwrap();
        assert!(bytes.is_none());
        assert_eq!(outcomes.len(), 1);
        match &outcomes[0] {
            BuildOutcome::ForcedColumnError { column, reason } => {
                assert_eq!(column, "does_not_exist");
                assert_eq!(reason, &SkipReason::MissingColumn);
            }
            _ => panic!("expected ForcedColumnError"),
        }
    }

    #[test]
    fn build_obs_predicate_index_bytes_preset_missing_warns() {
        let batch = make_obs_batch();
        let shard_ranges = vec![(0u64, 12u64)];
        let opts = PredicateIndexBuildOptions {
            forced_columns: vec![],
            preset_columns: vec!["cell_type".to_string(), "tissue".to_string()],
            auto_threshold: 1000,
            high_cardinality_threshold: 100_000,
        };
        let mut outcomes = Vec::new();
        let mut names = Vec::new();
        let bytes = build_obs_predicate_index_bytes(
            &batch,
            &shard_ranges,
            &opts,
            &mut outcomes,
            &mut names,
        )
        .unwrap();
        // cell_type exists in the fixture so an index is produced.
        assert!(bytes.is_some());
        assert_eq!(names, vec!["cell_type".to_string()]);
        // tissue is missing → preset skip with typed MissingColumn reason
        assert!(outcomes.iter().any(|o| matches!(
            o,
            BuildOutcome::PresetSkipped { column, reason }
                if column == "tissue" && *reason == SkipReason::MissingColumn
        )));
    }

    /// Forced column whose cardinality exceeds `high_cardinality_threshold`
    /// must surface as a `ForcedColumnError` with the typed
    /// `SkipReason::HighCardinality` discriminant. The Display impl is
    /// also exercised so callers re-using it for free-form messages
    /// stay stable.
    #[test]
    fn build_obs_predicate_index_bytes_forced_high_cardinality_errors() {
        // 100 rows, 100 unique values in `cell_id`. Setting
        // `high_cardinality_threshold = 10` is enough to reject it.
        let n_rows: usize = 100;
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n_rows).map(|i| format!("cell_{i}")).collect();
        let id_array = StringArray::from(ids);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(id_array)]).unwrap();
        let shard_ranges = vec![(0u64, n_rows as u64)];
        let opts = PredicateIndexBuildOptions {
            forced_columns: vec!["cell_id".to_string()],
            preset_columns: vec![],
            auto_threshold: 1000,
            high_cardinality_threshold: 10,
        };
        let mut outcomes = Vec::new();
        let mut names = Vec::new();
        let bytes = build_obs_predicate_index_bytes(
            &batch,
            &shard_ranges,
            &opts,
            &mut outcomes,
            &mut names,
        )
        .unwrap();
        // No column ended up indexed (the only forced one was rejected).
        assert!(bytes.is_none());
        assert!(names.is_empty());
        assert_eq!(outcomes.len(), 1);
        match &outcomes[0] {
            BuildOutcome::ForcedColumnError { column, reason } => {
                assert_eq!(column, "cell_id");
                match reason {
                    SkipReason::HighCardinality {
                        n_unique,
                        threshold,
                    } => {
                        assert_eq!(*n_unique, n_rows);
                        assert_eq!(*threshold, 10);
                    }
                    other => panic!("expected HighCardinality reason, got {other:?}"),
                }
                // Display impl should mention both numbers so callers
                // that stringify for warnings get useful output.
                let s = reason.to_string();
                assert!(s.contains("100"), "expected '100' in {s}");
                assert!(s.contains("10"), "expected '10' in {s}");
            }
            _ => panic!("expected ForcedColumnError"),
        }
    }
}
