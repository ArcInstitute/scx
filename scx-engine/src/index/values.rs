//! Arrow value extraction and type classification.
//!
//! The leaf layer: every builder above reaches an Arrow array through these,
//! and nothing here knows what a predicate index is.

use std::collections::{BTreeMap, HashSet};

use arrow::array::{
    Array, ArrayRef, AsArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;

use super::ShardRange;

/// Estimate the number of unique values in an Arrow array.
///
/// ⚠️ **Encoding-dependent, deliberately.** For a dictionary this is the
/// **declared** level count; for a plain column it is the count of values
/// actually present. So the same logical column can fall on either side of the
/// `auto_threshold` / `high_cardinality_threshold` cut depending on how it was
/// stored — a categorical with declared-but-unused levels looks bigger than
/// the plain column holding the same values. The consequence is only a loss of
/// pruning power (a column drops out of the auto-detected index), never a
/// wrong answer, and the dictionary branch is O(1) where the string branch
/// hashes every row.
///
/// Only the **batch-mode** builders reach this; the streaming builder
/// [`super::stream::ObsPredicateIndexBuilder`] accumulates observed values and
/// resolves cardinality at `finish`, so `merge` and `append` — which write
/// dictionaries — are unaffected. `scx upgrade --index-obs` over such a file is
/// the path that sees the declared count.
pub(crate) fn estimate_unique_values(col: &ArrayRef) -> usize {
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
/// [`super::stream::ObsPredicateIndexBuilder::push_shard`] to accept shards whose
/// per-shard upcast widens columns to `LargeUtf8` while the builder
/// was initialised with the input file's narrow `Utf8` schema.
pub(crate) fn column_class_compatible(a: &DataType, b: &DataType) -> bool {
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
pub(crate) fn logical_type(dt: &DataType) -> &DataType {
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
pub(crate) fn is_categorical_type(dt: &DataType) -> bool {
    matches!(logical_type(dt), DataType::Utf8 | DataType::LargeUtf8)
}

/// Check if a data type is numeric, looking through dictionary encoding
/// so an integer- or float-valued pandas `Categorical` is indexed as the
/// numbers it holds.
pub(crate) fn is_numeric_type(dt: &DataType) -> bool {
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
pub(crate) fn extract_string_value(col: &ArrayRef, row: usize) -> Option<String> {
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
pub(crate) fn extract_numeric_value(col: &ArrayRef, row: usize) -> Option<f64> {
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
pub(crate) fn rows_to_shard_ranges(
    rows: &[u64],
    shard_row_ranges: &[(u64, u64)],
) -> Vec<ShardRange> {
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
