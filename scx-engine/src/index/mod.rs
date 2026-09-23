//! Predicate index — the level-2 half of the two-level pushdown strategy
//! (docs/api.md (Query engine, optimizations)), split by concern.
//!
//! | Module | Holds |
//! |---|---|
//! | this one | the on-disk structs and `IndexKind`, plus the re-exports every `index::<name>` path resolves through |
//! | `wire` | the docs/format.md binary layout — `write_to` / `read_from`, v1 and v2 |
//! | `lookup` | query-time lookups: what the row-set fast path asks an index |
//! | `build` | index construction, eager, and the shared categorical / numeric primitives |
//! | `stream` | the streaming obs builder, for callers that never hold the whole `RecordBatch` |
//! | `derive` | per-shard catalog `column_stats` derived from a finished index (level 1) |
//! | `diagnostics` | CLI-facing message rendering, index presets, and the CSC policy default |
//! | `values` | Arrow value extraction and type classification, the leaf layer everything else builds on |
//!
//! The submodules are private — every name above is re-exported here, so the
//! module layout is an implementation detail and `index::<name>` stays the
//! import path. That is why the table above does not link them.

mod build;
mod derive;
mod diagnostics;
mod lookup;
mod stream;
mod values;
mod wire;

// `index::<name>` is an import path in four other crates — `resolve_csc_policy`
// and `SkipReason` from `pyscx`, `forced_columns_missing_message` from
// `scx-convert` and `scx-ops`, `IndexedColumn` from `scx-cli` — and eleven of
// these names are also flat-re-exported from `lib.rs`. The split has to keep
// every one of them resolving, so everything reachable at `index::<name>`
// before is re-exported here at the same visibility.
pub use build::{
    build_and_write_conversion_predicate_indexes,
    build_and_write_conversion_predicate_indexes_streaming, build_categorical_index, build_indexes,
    build_numeric_index, build_obs_predicate_index_bytes, build_var_predicate_index_bytes,
    resolve_predicate_index_build_options, BuildOutcome, ConversionPredicateIndexOptions,
    ConversionPredicateIndexResult, PredicateIndexBuildOptions, ResolvedIndexBuildOptions,
    SkipReason, HIGH_CARDINALITY_THRESHOLD,
};
pub use derive::{apply_obs_shard_column_stats, derive_shard_column_stats};
pub use diagnostics::{
    column_not_found_message, forced_column_missing_message, forced_columns_missing_message,
    index_preset_columns, resolve_csc_policy, IndexPreset,
};
pub use lookup::index_covers_all_obs;
pub use stream::ObsPredicateIndexBuilder;

// Reached only from `index_tests.rs`, which is attached to this module and so
// cannot see the submodules' items directly. Gated on `cfg(test)` so that a
// non-test caller of a helper widened purely for a test fails to compile
// rather than quietly acquiring a crate-wide dependency on it.
#[cfg(test)]
pub(crate) use stream::NUMERIC_BLOCK_ROWS;
#[cfg(test)]
pub(crate) use wire::requires_v2_encoding;

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
/// no value (see `build::numeric_leaves_from_spans`).
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

/// Which kind of index, if any, covers a column. Returned by
/// [`PredicateIndex::indexed_kind`] so the query partitioner can decide whether
/// a predicate on a column is resolvable from the index (row-set fast path) or
/// must fall back to decoding obs shards (residual path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    Categorical,
    Numeric,
}

#[cfg(test)]
#[path = "../index_tests.rs"]
mod tests;
