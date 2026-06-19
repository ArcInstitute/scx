//! `scx sort` — global obs-axis row reorder for query locality.
//!
//! This module holds the **shared sort core**:
//! the pure, reusable primitives every delivery form builds on —
//! option/result types, sort-key extraction, the stability comparator,
//! leading-key partition-boundary computation, the predicate-index-rebuild
//! feed, and the provenance entry.
//!
//! It deliberately contains **no I/O engine, CLI, or pyscx surface**. The
//! `sort()` engine + strategy selector, convert-gather,
//! k-way merge, and multimodal/obsp remap build on these
//! primitives.
//!
//! Design notes:
//! - Key extraction and comparison go through `arrow::row::RowConverter`,
//!   which handles categorical (`DictionaryArray` = codes + dictionary),
//!   numeric, composite (multi-column), descending (`--reverse`), and null
//!   ordering uniformly, producing byte-comparable `Row`s.
//! - The effective sort is on `(full_key, source_row_id)` so equal-key ties
//!   keep their original global order — stable and deterministic regardless
//!   of the order rows arrive in (e.g. Phase 4's parallel scatter).
//! - Phase 0 recorded the design decisions (bitmap drop-only for v1; obs
//!   axis globally shared; CSC rebuilt post-write via
//!   [`crate::rebuild_csc_inplace`]).

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::path::PathBuf;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::compute::SortOptions as ArrowSortOptions;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::row::{Row, RowConverter, Rows, SortField};
use scx_codec::CodecSelection;
use scx_engine::{
    build_and_write_conversion_predicate_indexes_streaming, ConversionPredicateIndexOptions,
    ConversionPredicateIndexResult,
};
use scx_format_io::{BitmapPolicy, ProvenanceEntry, ScxWriter, DEFAULT_SHARD_TARGET_ROWS};

use crate::error::{OpsError, Result};

// ---------------------------------------------------------------------------
// T1.1 — option / result types
// ---------------------------------------------------------------------------

/// Caller-supplied options for a sort (library core; the CLI layer parses
/// `--by` via `parse_index_columns` and constructs this).
#[derive(Debug, Clone)]
pub struct SortOptions {
    /// obs columns to sort by, lexicographic in `--by` order (leading key
    /// first). Empty is rejected at key-extractor construction.
    pub by: Vec<String>,
    /// Descending on all keys.
    pub reverse: bool,
    /// Output shard target rows.
    pub shard_target_rows: u32,
    /// Output codec selection (`Auto` defers per-shard selection to the
    /// writer; `Explicit(c)` forces codec `c`).
    pub codec: CodecSelection,
    /// Predicate-index rebuild options for the output. The sort key is
    /// auto-added to `index_obs` by [`rebuild_obs_predicate_index_streaming`].
    pub index_options: ConversionPredicateIndexOptions,
    /// Spill / partition memory budget in bytes (`None` = use defaults).
    pub memory_budget: Option<u64>,
    /// Spill location for the external sort (`None` = system temp).
    pub temp_dir: Option<PathBuf>,
    /// Detection-bitmap rebuild policy for the output (`Off` = drop, the
    /// default; `Auto`/`Always` rebuild the gene→local-row sidecar per X
    /// shard, mirroring `scx convert --bitmap`).
    pub bitmap: BitmapPolicy,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            by: Vec::new(),
            reverse: false,
            shard_target_rows: DEFAULT_SHARD_TARGET_ROWS,
            codec: CodecSelection::Auto,
            index_options: ConversionPredicateIndexOptions::default(),
            memory_budget: None,
            temp_dir: None,
            bitmap: BitmapPolicy::default(),
        }
    }
}

/// Which execution strategy actually ran (filled by the Phase 4 engine; the
/// build-time forms report their own variant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortStrategy {
    /// (a) In-memory argsort fast path.
    #[default]
    InMemory,
    /// (b) K-pass-by-category, zero spill (low-cardinality local input).
    KPassByCategory,
    /// (c) External partition sort (general / cloud / high-K).
    ExternalPartition,
    /// Build-time form 1: spill-free convert gather.
    ConvertGather,
    /// Build-time form 2: sorted k-way merge.
    KwayMerge,
}

/// Outcome of a sort.
#[derive(Debug, Clone, Default)]
pub struct SortSummary {
    /// Live rows written to the output.
    pub n_obs: u64,
    /// Number of output CSR shards.
    pub n_output_shards: u64,
    /// Strategy that ran.
    pub strategy: SortStrategy,
    /// Bytes spilled to scratch (0 for the spill-free paths).
    pub spill_bytes: u64,
    /// Number of leading-key partitions used.
    pub partitions: usize,
    /// obs columns for which a predicate index was (re)built.
    pub indexed_columns: Vec<String>,
    /// Whether the obs write took the memory-bounded spill-scatter path
    /// (`false` = the in-memory `take` path; always `false` for multimodal /
    /// legacy single-section obs / no `--memory-budget`).
    pub obs_spilled: bool,
    /// Number of obs `new_pos`-range spill partitions used (0 unless
    /// `obs_spilled`).
    pub obs_partitions: usize,
}

// ---------------------------------------------------------------------------
// T1.2 — key extraction
// ---------------------------------------------------------------------------

/// Extracts a comparable composite sort key per obs row.
///
/// Wraps an `arrow::row::RowConverter` built from the `--by` columns; one
/// `Row` is produced per obs row and encodes the full composite key with
/// `--reverse` and null ordering baked in.
pub struct SortKeyExtractor {
    converter: RowConverter,
    by: Vec<String>,
}

impl SortKeyExtractor {
    /// Build an extractor for the given obs `schema` and `by` columns.
    /// Errors if `by` is empty or names a column absent from `schema`.
    pub fn new(schema: &Schema, by: &[String], reverse: bool) -> Result<Self> {
        if by.is_empty() {
            return Err(OpsError::InvalidInput(
                "sort requires at least one --by column".to_string(),
            ));
        }
        let opts = ArrowSortOptions {
            descending: reverse,
            nulls_first: true,
        };
        let mut fields = Vec::with_capacity(by.len());
        for name in by {
            let field = schema.field_with_name(name).map_err(|_| {
                OpsError::InvalidInput(format!("sort key column '{name}' not found in obs schema"))
            })?;
            fields.push(SortField::new_with_options(field.data_type().clone(), opts));
        }
        let converter = RowConverter::new(fields)?;
        Ok(Self {
            converter,
            by: by.to_vec(),
        })
    }

    /// Convert `batch`'s key columns (in `--by` order) into byte-comparable
    /// `Rows` — one `Row` per row, the full composite key.
    pub fn rows(&self, batch: &RecordBatch) -> Result<Rows> {
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(self.by.len());
        for name in &self.by {
            let col = batch.column_by_name(name).cloned().ok_or_else(|| {
                OpsError::InvalidInput(format!("sort key column '{name}' missing from batch"))
            })?;
            cols.push(col);
        }
        Ok(self.converter.convert_columns(&cols)?)
    }
}

// ---------------------------------------------------------------------------
// T1.3 — comparator + stability
// ---------------------------------------------------------------------------

/// The stability rule: compare on the full key, breaking ties by the global
/// source row-id so equal-key rows keep their original order. Reusable by the
/// out-of-order parallel scatter (Phase 4).
pub fn cmp_keyed(a: (&Row, u64), b: (&Row, u64)) -> Ordering {
    a.0.cmp(b.0).then(a.1.cmp(&b.1))
}

/// Stable argsort of a batch's rows by the sort key, ties broken by global
/// source row-id (`base_row_id + local index`). Returns global source row-ids
/// in sorted order.
///
/// For a single in-order batch the stable sort already preserves ties; the
/// explicit id keeps the contract identical to the spilled / merged paths,
/// where rows do not arrive in source order.
pub fn stable_argsort(rows: &Rows, base_row_id: u64) -> Vec<u64> {
    let mut idx: Vec<usize> = (0..rows.num_rows()).collect();
    idx.sort_by(|&a, &b| rows.row(a).cmp(&rows.row(b)));
    idx.into_iter().map(|i| base_row_id + i as u64).collect()
}

// ---------------------------------------------------------------------------
// T1.4 — leading-key partition boundaries
// ---------------------------------------------------------------------------

/// Leading-key partition boundaries (external-sort pass 0). Always
/// computed on the **leading** key only; trailing keys are handled by the
/// within-partition sort.
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionPlan {
    /// One partition per group of adjacent categorical values, in sorted
    /// order. A single category larger than the target is its own group
    /// (Phase 4 sub-splits it at write time).
    Categorical { groups: Vec<Vec<String>> },
    /// Ascending numeric cut points; a value `v` falls in the first
    /// partition whose cut point is `> v`. `n` cut points → `n + 1`
    /// partitions.
    Numeric { cut_points: Vec<f64> },
}

impl PartitionPlan {
    /// Number of partitions this plan describes.
    pub fn n_partitions(&self) -> usize {
        match self {
            PartitionPlan::Categorical { groups } => groups.len(),
            PartitionPlan::Numeric { cut_points } => cut_points.len() + 1,
        }
    }
}

/// Target rows per partition from the memory budget and matrix shape
/// (one partition ≈ a budget's worth of decoded rows). `None` budget → one
/// shard's worth. The result
/// is floored at `shard_target_rows` so a partition is always at least one
/// output shard.
pub fn partition_target_rows(
    memory_budget: Option<u64>,
    n_vars: usize,
    density: f64,
    shard_target_rows: u32,
) -> usize {
    let floor = (shard_target_rows.max(1)) as usize;
    match memory_budget {
        Some(budget) => {
            let per_row = (n_vars as f64 * density * 16.0).max(1.0);
            let rows = (budget as f64 / per_row).floor() as usize;
            rows.max(floor)
        }
        None => floor,
    }
}

/// Numeric DataTypes the leading-key numeric path handles (everything else is
/// treated as categorical / string-like).
fn is_numeric_dtype(dt: &DataType) -> bool {
    use DataType::*;
    matches!(
        dt,
        Int8 | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Float16
            | Float32
            | Float64
    )
}

/// Distinct categorical values + counts in ascending value order. Casts
/// `DictionaryArray` / `LargeUtf8` to `Utf8` so all string-like encodings are
/// handled uniformly. Null values are excluded from boundary computation
/// (Phase 4 routes nulls to the leading partition under `nulls_first`).
fn category_counts(col: &ArrayRef) -> Result<Vec<(String, u64)>> {
    let utf8 = arrow::compute::cast(col, &DataType::Utf8)?;
    let arr = utf8.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        OpsError::InvalidInput(format!(
            "leading sort key dtype {:?} is not categorical/string-like",
            col.data_type()
        ))
    })?;
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for i in 0..arr.len() {
        if arr.is_null(i) {
            continue;
        }
        *counts.entry(arr.value(i).to_string()).or_insert(0) += 1;
    }
    Ok(counts.into_iter().collect())
}

/// Group adjacent (sorted) categories so each group holds at most
/// `target_rows` rows. A single category exceeding the target becomes its own
/// group.
fn group_sorted_categories(sorted: Vec<(String, u64)>, target_rows: usize) -> Vec<Vec<String>> {
    let target = target_rows.max(1) as u64;
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut cur_count: u64 = 0;
    for (val, cnt) in sorted {
        if !cur.is_empty() && cur_count + cnt > target {
            groups.push(std::mem::take(&mut cur));
            cur_count = 0;
        }
        cur.push(val);
        cur_count += cnt;
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

/// Build an equal-width histogram over `values` in `[min, max]`. Returns
/// `(bin_upper_bound, count)` pairs in ascending order. A degenerate range
/// (`max <= min`) collapses to a single bin.
pub fn numeric_histogram(
    values: impl Iterator<Item = f64>,
    min: f64,
    max: f64,
    n_bins: usize,
) -> Vec<(f64, u64)> {
    let n_bins = n_bins.max(1);
    if max <= min {
        return vec![(min, values.count() as u64)];
    }
    let width = (max - min) / n_bins as f64;
    let mut counts = vec![0u64; n_bins];
    for v in values {
        let mut b = ((v - min) / width).floor() as isize;
        if b < 0 {
            b = 0;
        }
        if b >= n_bins as isize {
            b = n_bins as isize - 1;
        }
        counts[b as usize] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, c)| (min + (i as f64 + 1.0) * width, c))
        .collect()
}

/// Derive ascending cut points from a histogram so each partition holds
/// ≈`target_rows`. Never emits a cut on the final bin (that would create an
/// empty trailing partition).
pub fn numeric_cut_points(hist: &[(f64, u64)], target_rows: usize) -> Vec<f64> {
    let target = target_rows.max(1) as u64;
    let mut cuts = Vec::new();
    let mut running: u64 = 0;
    let last = hist.len().saturating_sub(1);
    for (i, (upper, cnt)) in hist.iter().enumerate() {
        running += cnt;
        if running >= target && i != last {
            cuts.push(*upper);
            running = 0;
        }
    }
    cuts
}

/// Compute the leading-key partition plan for a column. Numeric leading keys
/// use a 256-bin histogram + quantile cut points; everything else is treated
/// categorically (sorted distinct values grouped to `target_rows`).
pub fn leading_key_partitions(
    col: &ArrayRef,
    target_rows: usize,
    reverse: bool,
) -> Result<PartitionPlan> {
    if is_numeric_dtype(col.data_type()) {
        let casted = arrow::compute::cast(col, &DataType::Float64)?;
        let arr = casted
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                OpsError::InvalidInput("failed to cast numeric sort key to f64".to_string())
            })?;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut n = 0u64;
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                let v = arr.value(i);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                n += 1;
            }
        }
        if n == 0 {
            return Ok(PartitionPlan::Numeric { cut_points: vec![] });
        }
        let hist = numeric_histogram(
            (0..arr.len())
                .filter(|&i| !arr.is_null(i))
                .map(|i| arr.value(i)),
            min,
            max,
            256,
        );
        Ok(PartitionPlan::Numeric {
            cut_points: numeric_cut_points(&hist, target_rows),
        })
    } else {
        let mut counts = category_counts(col)?;
        if reverse {
            counts.reverse();
        }
        Ok(PartitionPlan::Categorical {
            groups: group_sorted_categories(counts, target_rows),
        })
    }
}

// ---------------------------------------------------------------------------
// T1.5 — predicate-index-rebuild feed
// ---------------------------------------------------------------------------

/// Rebuild the obs predicate index from a stream of `(shard_batch,
/// shard_row_offset)` in ascending row order, auto-adding the sort key
/// columns to `index_obs` so their (now contiguous) `shard_ranges` are
/// written. Thin wrapper over
/// [`build_and_write_conversion_predicate_indexes_streaming`].
///
/// Callers feed pass-2's emitted obs shards directly; the underlying builder
/// asserts `shard_row_offset == rows_pushed`, so shards must be in order.
///
/// Items are `Result`-typed so a fallible producer (e.g. the obs spill-scatter
/// reader, which decodes shards lazily off disk) can surface I/O errors mid
/// stream; the in-memory caller wraps its single batch in `Ok`.
// Thin pass-through over the 7-arg engine builder plus the sort-key list;
// bundling these into a struct would only obscure the 1:1 mapping.
#[allow(clippy::too_many_arguments)]
pub fn rebuild_obs_predicate_index_streaming<I>(
    writer: &mut ScxWriter,
    obs_schema: SchemaRef,
    obs_shards: I,
    var: &RecordBatch,
    obs_row_ranges: &[(u64, u64)],
    n_vars: usize,
    by: &[String],
    base_options: &ConversionPredicateIndexOptions,
) -> Result<ConversionPredicateIndexResult>
where
    I: IntoIterator<Item = std::result::Result<(RecordBatch, u64), scx_engine::EngineError>>,
{
    let mut options = base_options.clone();
    for key in by {
        if !options.index_obs.iter().any(|c| c == key) {
            options.index_obs.push(key.clone());
        }
    }
    let result = build_and_write_conversion_predicate_indexes_streaming(
        writer,
        obs_schema,
        obs_shards,
        var,
        obs_row_ranges,
        n_vars,
        &options,
    )?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// T1.6 — provenance
// ---------------------------------------------------------------------------

/// Build the `sort` provenance entry (mirrors the `compact` entry shape). The
/// caller supplies `timestamp` (Unix seconds) so tests stay deterministic;
/// the Phase-4 caller passes `SystemTime::now()`.
pub fn sort_provenance_entry(
    by: &[String],
    reverse: bool,
    shard_target_rows: u32,
    indexed_columns: &[String],
    timestamp: i64,
) -> ProvenanceEntry {
    let params = serde_json::json!({
        "by": by,
        "reverse": reverse,
        "shard_size": shard_target_rows,
        "predicate_index": { "obs_columns": indexed_columns },
    });
    ProvenanceEntry {
        timestamp,
        action: "sort".to_string(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: params.to_string(),
        input_checksums: vec![],
    }
}

#[cfg(test)]
#[path = "sort_tests.rs"]
mod tests;
