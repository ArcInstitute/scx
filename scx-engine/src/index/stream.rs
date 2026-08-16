//! Streaming obs predicate-index builder.
//!
//! The single-batch entry point in [`super::build`] requires the whole obs
//! `RecordBatch` in memory. This builder accepts one shard at a time and holds
//! only the per-column accumulators, so a merge or append that never
//! materialises obs can still emit an index.

use std::collections::BTreeMap;

use arrow::array::Array;

use crate::error::{EngineError, Result};

use super::build::{
    numeric_index_from_spans, ranges_ascending_disjoint, BuildOutcome, NumericSpan,
    PredicateIndexBuildOptions, SkipReason,
};
use super::values::{
    column_class_compatible, extract_numeric_value, extract_string_value, is_categorical_type,
    is_numeric_type, rows_to_shard_ranges,
};
use super::{CategoricalEntry, CategoricalIndex, IndexedColumn, PredicateIndex};

/// Streaming counterpart to [`super::build::build_obs_predicate_index_bytes`].
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
pub(crate) const NUMERIC_BLOCK_ROWS: u64 = 1024;

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
    /// [`super::build::build_obs_predicate_index_bytes`]). `indexed_column_names`
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
