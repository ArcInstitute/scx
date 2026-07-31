//! Streaming `(codes, categories)` extraction for string/categorical obs
//! columns, shared by the local ([`crate::reader::ScxReader`]) and cloud
//! (`scx_cloud::CloudReader`) read paths.
//!
//! This is the numpy-level counterpart to [`crate::distinct`]: where
//! `DistinctAccumulator` answers "what values exist" from the per-shard
//! dictionary catalogs alone, [`GlobalCategoryAccum`] also produces the
//! per-row **codes**, which is what a consumer building a global one-hot /
//! vocabulary map actually wants.
//!
//! ## Why a dedicated accumulator rather than `read_obs_keys`
//!
//! `read_obs_keys` → `concat_batches` → `unify_dictionary_columns` also yields a
//! dictionary-encoded column with a unified vocabulary, but it materialises the
//! whole concatenated column first — the multi-GB-per-column transient that
//! motivated the dictionary-values dedup in the first place. This accumulator
//! folds **one shard at a time** and keeps only the running vocabulary plus the
//! output codes, so peak memory is `n_obs × 4 B` for the codes and O(distinct)
//! for the vocabulary, with no per-shard column ever retained.
//!
//! ## Semantics (pandas-compatible)
//!
//! - **Null → code `-1`**, matching `pandas.Categorical.codes` and the h5ad
//!   codes convention. This deliberately differs from
//!   `scx_loader::decode_stage::CategoryDict`, which appends a synthetic
//!   trailing `"NaN"` level; that type is training-internal and documents that
//!   it is *not* `pandas.Categorical.codes`-stable. Keying on validity also
//!   keeps a genuine `"NaN"` *string* category distinct from a missing value.
//! - **Category order is first-seen**, scanning shards in index order and, for a
//!   dictionary shard, its declared dictionary order. Stable for a given file;
//!   not guaranteed to match a pandas `astype("category")` lexicographic order.
//! - **Unreferenced dictionary entries are preserved.** A non-compact shard
//!   dictionary contributes its declared categories even if no row uses them,
//!   which matches `astype("category")` retaining unused levels and matches
//!   [`crate::distinct`]'s documented superset behaviour.
//! - **Both on-disk representations are accepted** — `Dictionary(_, Utf8)` as
//!   `from_anndata` writes, and plain `Utf8`/`LargeUtf8` as `append` writes
//!   (which decodes dictionaries via `scx_ops::unify_dict_columns`). A file
//!   grown by `append` therefore carries *both* across its shards, and this
//!   folds them into one vocabulary without the
//!   `reconcile_dictionary_representations` cast the assembling path needs.
//!
//! ## Cardinality is unbounded, by necessity
//!
//! The running vocabulary has no cap. Unlike [`crate::distinct`] — whose `limit`
//! exists because enumerating *some* values is still a useful answer — a partial
//! code mapping is not: every row needs a code, so truncating the vocabulary would
//! silently mis-code rows. A pathological column (free text, or an accidentally
//! non-categorical barcode) therefore costs O(distinct) memory here, the same
//! exposure `read_obs` already has for that column. Callers that cannot trust a
//! column's cardinality should probe it first with
//! [`crate::reader::ScxReader::distinct_obs_values`], which *can* stop early.
//!
//! The interning strategy is lifted from `scx-convert`'s streaming h5ad
//! categorical writer (`h5ad/write.rs`'s `CatAccum` / `local_categorical_view`),
//! which solves the same problem on the write side. It is duplicated rather than
//! shared because `scx-convert` is `hdf5`-feature-gated and this must build on a
//! default `cargo test`.

use std::collections::HashMap;

use arrow::array::{Array, ArrayRef, AsArray, GenericStringArray, OffsetSizeTrait};
use arrow::datatypes::DataType;

use crate::error::{Result, ScxError};

/// Folds per-shard obs column arrays into one global `(codes, categories)` pair.
///
/// See the [module docs](self) for null / ordering / representation semantics.
pub struct GlobalCategoryAccum {
    column: String,
    /// category → global code.
    dict: HashMap<String, i32>,
    /// global code → category (index *is* the code).
    order: Vec<String>,
    /// Per-row global codes in global obs row order; `-1` for null.
    codes: Vec<i32>,
}

impl GlobalCategoryAccum {
    /// Create an accumulator for `column`. `n_obs_hint` pre-sizes the codes
    /// buffer; a wrong hint costs only a reallocation.
    pub fn new(column: impl Into<String>, n_obs_hint: usize) -> Self {
        Self {
            column: column.into(),
            dict: HashMap::new(),
            order: Vec::new(),
            codes: Vec::with_capacity(n_obs_hint),
        }
    }

    /// Fold one shard's projected column array, appending its rows' global codes.
    ///
    /// Shards must be pushed in **global row order** — the codes buffer is
    /// append-only, so out-of-order shards would silently misalign codes against
    /// obs rows. Both callers sort catalog entries by shard index first.
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
                // Intern the shard's whole declared vocabulary up front. This
                // both preserves unreferenced levels (see the module docs) and
                // gives an O(1) local→global lookup per row below.
                let local_to_global = match value_type.as_ref() {
                    DataType::Utf8 => self.intern_values(values.as_string::<i32>())?,
                    DataType::LargeUtf8 => self.intern_values(values.as_string::<i64>())?,
                    other => return Err(self.unsupported(other)),
                };
                // `normalized_keys` widens any key width (Int8/Int16/Int32) to
                // usize and reports null keys, so this arm is key-width agnostic.
                let keys = dict.normalized_keys();
                for (i, &local) in keys.iter().enumerate() {
                    if dict.is_null(i) {
                        self.codes.push(-1);
                        continue;
                    }
                    let g = *local_to_global.get(local).ok_or_else(|| {
                        ScxError::InvalidCatalog(format!(
                            "obs column '{}': dictionary key {local} out of range \
                             for a {}-entry dictionary",
                            self.column,
                            local_to_global.len()
                        ))
                    })?;
                    self.codes.push(g);
                }
            }
            DataType::Utf8 => self.ingest_plain(array.as_string::<i32>())?,
            DataType::LargeUtf8 => self.ingest_plain(array.as_string::<i64>())?,
            other => return Err(self.unsupported(other)),
        }
        Ok(())
    }

    /// Intern every value of a shard dictionary, returning `local → global`.
    fn intern_values<O: OffsetSizeTrait>(
        &mut self,
        values: &GenericStringArray<O>,
    ) -> Result<Vec<i32>> {
        let mut map = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            // A null *dictionary entry* (as opposed to a null key) cannot be
            // addressed by a valid key in any well-formed array; map it to -1 so
            // a malformed file degrades to "missing" instead of shifting codes.
            if values.is_null(i) {
                map.push(-1);
                continue;
            }
            map.push(self.intern(values.value(i))?);
        }
        Ok(map)
    }

    /// Fold a plain (non-dictionary) string shard, interning per row.
    fn ingest_plain<O: OffsetSizeTrait>(&mut self, arr: &GenericStringArray<O>) -> Result<()> {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                self.codes.push(-1);
                continue;
            }
            let g = self.intern(arr.value(i))?;
            self.codes.push(g);
        }
        Ok(())
    }

    /// Global code for `value`, assigning the next one on first sight.
    fn intern(&mut self, value: &str) -> Result<i32> {
        if let Some(&g) = self.dict.get(value) {
            return Ok(g);
        }
        let g: i32 = self.order.len().try_into().map_err(|_| {
            ScxError::InvalidCatalog(format!(
                "obs column '{}' has more than {} distinct categories",
                self.column,
                i32::MAX
            ))
        })?;
        self.dict.insert(value.to_string(), g);
        self.order.push(value.to_string());
        Ok(g)
    }

    fn unsupported(&self, dtype: &DataType) -> ScxError {
        ScxError::UnsupportedColumnType {
            column: self.column.clone(),
            dtype: format!("{dtype:?}"),
        }
    }

    /// Number of rows folded so far. Callers cross-check this against the file's
    /// physical `n_obs` to catch a shard-cover gap.
    pub fn n_rows(&self) -> usize {
        self.codes.len()
    }

    /// Consume into `(codes, categories)`.
    pub fn finish(self) -> (Vec<i32>, Vec<String>) {
        (self.codes, self.order)
    }
}

#[cfg(test)]
#[path = "categorical_tests.rs"]
mod tests;
