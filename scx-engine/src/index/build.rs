//! Index construction: the eager builders and the primitives the streaming
//! builder in [`super::stream`] shares with them.
//!
//! [`build_indexes`] is the low-level entry point; the `*_bytes` functions add
//! the forced/preset column policy the conversion pipeline needs; and
//! [`build_categorical_index`] / [`build_numeric_index`] / the `*_from_spans`
//! helpers are called by *both* this module and [`super::stream`], which is why
//! they live on this side of the split.

use std::collections::BTreeMap;

use arrow::array::{Array, ArrayRef};

use crate::error::{EngineError, Result};

use super::derive::apply_obs_shard_column_stats;
use super::diagnostics::index_preset_columns;
use super::stream::ObsPredicateIndexBuilder;
use super::values::{
    estimate_unique_values, extract_numeric_value, extract_string_value, is_categorical_type,
    is_numeric_type, rows_to_shard_ranges,
};
use super::{
    CategoricalEntry, CategoricalIndex, IndexedColumn, InternalPage, LeafPage, NumericIndex,
    NumericLeafEntry, PredicateIndex,
};

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

/// Are the shard ranges each well-formed *and* ascending and disjoint?
///
/// Both halves matter. Checking only `end <= next.start` accepts an inverted
/// range like `[(100, 50), (60, 70)]`, and the scans below would then stop at
/// the inverted entry's `start` and miss the valid overlap behind it — the
/// exhaustive fallback exists precisely so a malformed table cannot lose an
/// overlap, so the predicate that selects it has to be the stronger one.
/// Every in-tree table is catalog-derived and valid; this is about the public
/// helper's stated contract, not about a reachable in-tree bug.
pub(crate) fn ranges_ascending_disjoint(ranges: &[(u64, u64)]) -> bool {
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
pub(crate) struct NumericSpan {
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) first_row: u64,
    pub(crate) last_row: u64,
}

/// Fold row-ordered spans into **at most one [`NumericLeafEntry`] per shard** —
/// a shard no valued span reaches emits none, which is what keeps a null tail
/// out of the coverage calculation.
///
/// Per-shard is the granularity the leaves are actually consumed at:
/// [`super::derive::derive_shard_column_stats`] folds them to a per-shard
/// `ColumnStat::MinMax` (what Level-1 pruning reads) and
/// [`PredicateIndex::max_covered_global_row`] takes the per-shard maximum
/// `row_end` (what [`super::lookup::index_covers_all_obs`] reads). No query path ever looks
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
pub(crate) fn numeric_index_from_spans(
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

/// Hard cap above which a *named* (forced or preset) column is rejected as
/// unsupported. **The one derivation site**: every builder in the workspace
/// reaches this constant through [`resolve_predicate_index_build_options`] or
/// through the `scx-ops` pass built on it. It used to be a `100_000` literal
/// spelled out at seven call sites — `streaming_impl` here plus six in
/// `scx-ops` — and a CI guard now rejects a new one (ORG-6.14-2).
///
/// The cap exists to keep a forced/preset column from blowing up the index;
/// auto-detection uses `ConversionPredicateIndexOptions::index_auto_threshold`
/// (default 1000) instead.
pub const HIGH_CARDINALITY_THRESHOLD: usize = 100_000;

/// The obs and var halves of a resolved predicate-index request.
///
/// Produced once by [`resolve_predicate_index_build_options`] so the preset
/// lookup, the forced-column list, the auto threshold and
/// [`HIGH_CARDINALITY_THRESHOLD`] cannot be spelled differently on the two
/// axes or by two callers.
#[derive(Debug, Clone)]
pub struct ResolvedIndexBuildOptions {
    /// Options for the obs axis.
    pub obs: PredicateIndexBuildOptions,
    /// Options for the var axis.
    pub var: PredicateIndexBuildOptions,
}

/// Resolve a [`ConversionPredicateIndexOptions`] into the per-axis
/// [`PredicateIndexBuildOptions`] the builders take.
///
/// **Errors on an unknown `index_preset`**, and does so before the caller has
/// touched any writer state — which is the whole reason this is a separate,
/// fallible step rather than something each builder does inline. Six `scx-ops`
/// call sites used to resolve the preset with
/// `index_preset_columns(name).map(…).unwrap_or_default()`, so
/// `scx merge --index-preset typo` built no preset columns and exited 0 while
/// `scx convert --index-preset typo` failed. They now share this function and
/// fail the same way.
pub fn resolve_predicate_index_build_options(
    options: &ConversionPredicateIndexOptions,
) -> Result<ResolvedIndexBuildOptions> {
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
    Ok(ResolvedIndexBuildOptions {
        obs: PredicateIndexBuildOptions {
            forced_columns: options.index_obs.clone(),
            preset_columns: preset_obs,
            auto_threshold: options.index_auto_threshold,
            high_cardinality_threshold: HIGH_CARDINALITY_THRESHOLD,
        },
        var: PredicateIndexBuildOptions {
            forced_columns: options.index_var.clone(),
            preset_columns: preset_var,
            auto_threshold: options.index_auto_threshold,
            high_cardinality_threshold: HIGH_CARDINALITY_THRESHOLD,
        },
    })
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
    // Resolve preset up front so an unknown name fails before any
    // writer state changes.
    let resolved = resolve_predicate_index_build_options(options)?;

    let mut result = ConversionPredicateIndexResult::default();

    // obs — stream shards through the builder.
    let mut builder = ObsPredicateIndexBuilder::new(obs_schema, &resolved.obs)?;
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
    let var_bytes = build_var_predicate_index_bytes(
        var,
        &var_row_ranges,
        &resolved.var,
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
