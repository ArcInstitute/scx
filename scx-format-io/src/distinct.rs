//! Streaming distinct-value extraction for a single string/categorical obs
//! column, shared by the local ([`crate::reader::ScxReader`]) and cloud
//! (`scx_cloud::CloudReader`) read paths.
//!
//! The accumulator folds one shard's projected column array at a time, so a
//! caller never has to assemble the full multi-shard obs table (which is the
//! expensive [`crate::reader::assemble_sharded_metadata`] path). For
//! dictionary-encoded columns it scans only the per-shard dictionary *values*
//! (the catalog) — the row keys are never touched, making low-cardinality
//! enumeration nearly free.
//!
//! ## Correctness caveats (mirror the pyscx docstring)
//!
//! - **Dictionary fast path returns a superset.** An Arrow dictionary column
//!   can carry dictionary entries that no row references (a non-compact
//!   dictionary). Unioning per-shard dictionaries therefore yields distinct
//!   *candidate* values that may include unreferenced categories. For
//!   `from_anndata`-written categoricals this is normally compact, but it is
//!   not guaranteed.
//! - **Nulls are excluded.** Missing values never appear in the result; this
//!   is the vocabulary-enumeration contract.
//! - **`limit` without `sort` is "first N encountered."** Cheap: scanning
//!   stops as soon as an `(N+1)`-th distinct value is observed, and `has_more`
//!   is set. With `sort = true` the short-circuit is disabled — all distinct
//!   values are collected, sorted, then truncated to `limit`.

use std::collections::HashSet;

use arrow::array::{Array, ArrayRef, AsArray, GenericStringArray, OffsetSizeTrait};
use arrow::datatypes::DataType;

use crate::error::{Result, ScxError};

/// Ordered, null-excluded accumulator of the distinct string values of one obs
/// column, folded shard-by-shard. See the [module docs](self) for the
/// dictionary-superset / null / `limit` semantics.
pub struct DistinctAccumulator {
    column: String,
    seen: HashSet<String>,
    /// Distinct values in first-encountered order (across shards, then within
    /// each shard's value buffer).
    ordered: Vec<String>,
    limit: Option<usize>,
    sort: bool,
    /// Set once an `(N+1)`-th distinct value is seen under a `limit` (and
    /// `sort == false`); also computed in [`Self::finish`] for the sorted path.
    has_more: bool,
}

impl DistinctAccumulator {
    /// Create an accumulator for `column`. `limit` caps the returned set;
    /// `sort` collects everything and sorts before truncating (disables the
    /// first-N short-circuit).
    pub fn new(column: impl Into<String>, limit: Option<usize>, sort: bool) -> Self {
        Self {
            column: column.into(),
            seen: HashSet::new(),
            ordered: Vec::new(),
            limit,
            sort,
            has_more: false,
        }
    }

    /// The effective number of distinct values to collect before switching to
    /// overflow-detection. Infinite when no limit is set or when sorting (the
    /// sorted path needs every value before it can truncate).
    fn collect_cap(&self) -> usize {
        if self.sort {
            usize::MAX
        } else {
            self.limit.unwrap_or(usize::MAX)
        }
    }

    /// Fold one shard's projected column array into the distinct set.
    ///
    /// For `Dictionary(_, Utf8|LargeUtf8)` only the dictionary values are
    /// scanned (any key width — `Int8`/`Int16`/`Int32`). For plain
    /// `Utf8`/`LargeUtf8` the value buffer is scanned directly. Any other dtype
    /// is rejected as [`ScxError::UnsupportedColumnType`].
    pub fn push(&mut self, array: &ArrayRef) -> Result<()> {
        match array.data_type() {
            DataType::Dictionary(_, value_type) => {
                let dict = array.as_any_dictionary_opt().ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "obs column '{}' declared Dictionary but failed downcast",
                        self.column
                    ))
                })?;
                let values = dict.values();
                match value_type.as_ref() {
                    DataType::Utf8 => self.ingest_strings(values.as_string::<i32>()),
                    DataType::LargeUtf8 => self.ingest_strings(values.as_string::<i64>()),
                    other => return Err(self.unsupported(other)),
                }
            }
            DataType::Utf8 => self.ingest_strings(array.as_string::<i32>()),
            DataType::LargeUtf8 => self.ingest_strings(array.as_string::<i64>()),
            other => return Err(self.unsupported(other)),
        }
        Ok(())
    }

    /// Scan one string array, interning distinct non-null values. Stops early
    /// (setting `has_more`) once a distinct value beyond `collect_cap()` is
    /// found and `sort == false`.
    fn ingest_strings<O: OffsetSizeTrait>(&mut self, arr: &GenericStringArray<O>) {
        let cap = self.collect_cap();
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            let v = arr.value(i);
            if self.ordered.len() >= cap {
                // First-N already collected; we only need to know whether any
                // further distinct value exists.
                if !self.seen.contains(v) {
                    self.has_more = true;
                    return;
                }
                continue;
            }
            if self.seen.insert(v.to_string()) {
                self.ordered.push(v.to_string());
            }
        }
    }

    /// True once no further data needs to be scanned: a `limit` is set, we are
    /// not sorting, and the overflow value has already been observed.
    pub fn done(&self) -> bool {
        !self.sort && self.has_more
    }

    fn unsupported(&self, dtype: &DataType) -> ScxError {
        ScxError::UnsupportedColumnType {
            column: self.column.clone(),
            dtype: format!("{dtype:?}"),
        }
    }

    /// Consume the accumulator, returning `(values, has_more)`. For the sorted
    /// path, values are sorted and truncated to `limit` (with `has_more`
    /// reflecting whether truncation dropped any).
    pub fn finish(mut self) -> (Vec<String>, bool) {
        if self.sort {
            self.ordered.sort();
            if let Some(n) = self.limit {
                if self.ordered.len() > n {
                    self.ordered.truncate(n);
                    return (self.ordered, true);
                }
            }
            (self.ordered, false)
        } else {
            (self.ordered, self.has_more)
        }
    }
}

#[cfg(test)]
#[path = "distinct_tests.rs"]
mod tests;
