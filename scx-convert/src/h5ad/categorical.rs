//! Arrow dictionary → h5ad categorical: decode, widen, and accumulate.
//!
//! Both dataframe-writing modules need this and neither owns it, which is why
//! it is its own file rather than living in one of them: [`super::columns`]'s
//! layout pre-pass calls [`local_categorical_view`] to learn which categories
//! a shard actually uses, and [`super::column_stream`] calls it again on the
//! write pass to remap local codes onto the global vocabulary held in
//! [`CatAccum`]. The `pub(super)` items are exactly that shared surface.
//!
//! Distinct from `scx-format-io`'s `categorical` module, which is about the
//! SCX-side wire encoding; this one is about h5ad's.

use crate::h5_write_util::vlu;
use crate::pipeline::ConvertError;
use arrow::array::{Array, AsArray, DictionaryArray};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use hdf5::types::VarLenUnicode;
use std::collections::HashMap;

pub(super) fn downcast_err(name: &str, expected: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': expected {expected} array but downcast failed"
    ))
}

/// Extract categorical codes promoted to i32 (-1 for null). Handles all
/// integer key widths Arrow / pandas uses (Int8/16/32/64, UInt8/16/32/64).
/// The SCX → h5ad writer emits i32 codes uniformly so anndata's reader
/// doesn't need to dispatch on key width.
fn dict_codes_i32(array: &dyn Array, name: &str) -> Result<Vec<i32>, ConvertError> {
    macro_rules! codes {
        ($t:ty, $label:literal) => {{
            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<$t>>()
                .ok_or_else(|| downcast_err(name, $label))?;
            Ok(dict
                .keys()
                .iter()
                .map(|v| match v {
                    Some(k) => k as i32,
                    None => -1,
                })
                .collect())
        }};
    }
    let key_type = match array.data_type() {
        DataType::Dictionary(k, _) => k.as_ref(),
        _ => {
            return Err(ConvertError::Other(format!(
                "column '{name}': expected Dictionary, got {:?}",
                array.data_type()
            )))
        }
    };
    match key_type {
        DataType::Int8 => codes!(Int8Type, "Dictionary<Int8, _>"),
        DataType::Int16 => codes!(Int16Type, "Dictionary<Int16, _>"),
        DataType::Int32 => codes!(Int32Type, "Dictionary<Int32, _>"),
        DataType::Int64 => codes!(Int64Type, "Dictionary<Int64, _>"),
        DataType::UInt8 => codes!(UInt8Type, "Dictionary<UInt8, _>"),
        DataType::UInt16 => codes!(UInt16Type, "Dictionary<UInt16, _>"),
        DataType::UInt32 => codes!(UInt32Type, "Dictionary<UInt32, _>"),
        DataType::UInt64 => codes!(UInt64Type, "Dictionary<UInt64, _>"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': unsupported categorical key type {other:?}"
        ))),
    }
}

/// Categorical category payload, carrying the source value class so the
/// h5ad writer emits a `categories` dataset of the matching HDF5 dtype.
/// String categories are the common case; integer/float categories let
/// numeric-keyed categoricals (integer cluster labels, dose levels)
/// round-trip instead of being dropped.
pub(super) enum CatValues {
    Str(Vec<VarLenUnicode>),
    Int(Vec<i64>),
    Float(Vec<f64>),
}

/// Extract a categorical's category values, dispatching on the dictionary
/// value type (independent of the key width). Integer/unsigned widths
/// normalize to `i64`, floats to `f64`. Categories never carry nulls (the
/// codes carry NA via `-1`), so `value(i)` is always valid.
fn dict_category_values(array: &dyn Array, name: &str) -> Result<CatValues, ConvertError> {
    // `as_any_dictionary` panics on a non-dictionary array; callers only reach
    // here inside a `Dictionary(_, _)` match arm, but use the fallible form so
    // a wrong call surfaces a structured error instead of a panic.
    let dict = array.as_any_dictionary_opt().ok_or_else(|| {
        ConvertError::Other(format!(
            "column '{name}': expected Dictionary, got {:?}",
            array.data_type()
        ))
    })?;
    let values = dict.values();
    // Categories never carry nulls (the codes carry NA via `-1`). Iterate the
    // backing buffer / string array directly rather than the bounds-checked
    // `value(i)`.
    macro_rules! ints {
        ($t:ty) => {{
            let a = values.as_primitive::<$t>();
            let mut out = Vec::with_capacity(a.len());
            for &v in a.values().iter() {
                out.push(cat_int_to_i64(v, name)?);
            }
            CatValues::Int(out)
        }};
    }
    macro_rules! floats {
        ($t:ty) => {{
            let a = values.as_primitive::<$t>();
            CatValues::Float(a.values().iter().map(|&v| v as f64).collect())
        }};
    }
    Ok(match values.data_type() {
        DataType::Utf8 => {
            let a = values.as_string::<i32>();
            CatValues::Str(a.iter().map(|v| vlu(v.unwrap_or(""))).collect())
        }
        DataType::LargeUtf8 => {
            let a = values.as_string::<i64>();
            CatValues::Str(a.iter().map(|v| vlu(v.unwrap_or(""))).collect())
        }
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt32 => ints!(UInt32Type),
        DataType::UInt64 => ints!(UInt64Type),
        DataType::Float32 => floats!(Float32Type),
        DataType::Float64 => floats!(Float64Type),
        other => {
            return Err(ConvertError::Other(format!(
                "column '{name}': unsupported dictionary value type {other:?}"
            )));
        }
    })
}

pub(super) fn cardinality_err(name: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': categorical cardinality exceeds i32::MAX"
    ))
}

/// Normalize a categorical integer value to `i64`, **rejecting** an unsigned
/// value above `i64::MAX` rather than silently wrapping it to a negative label
/// (which would corrupt the category on round-trip). Signed widths and unsigned
/// widths ≤ 32 bits always fit, so only `UInt64` can actually fail; the checked
/// form is applied uniformly across widths (and on both the dictionary and the
/// plain-shard paths) so the two stay symmetric. The h5ad integer-categorical
/// `categories` dataset is `i64`, so out-of-range unsigned labels are
/// genuinely unrepresentable and must error rather than mis-encode.
fn cat_int_to_i64<T>(v: T, name: &str) -> Result<i64, ConvertError>
where
    T: TryInto<i64> + std::fmt::Display + Copy,
{
    v.try_into().map_err(|_| {
        ConvertError::Other(format!(
            "column '{name}': categorical integer value {v} exceeds the i64 range \
             of the h5ad integer-categorical encoding"
        ))
    })
}

/// Build a per-shard local categorical view `(local_codes, local_values)` for
/// the streaming categorical writer, accepting **both** a `Dictionary(_, V)`
/// array (what every write door but `scx sort`'s spill path now emits for a
/// categorical) and a **plain
/// `V`** array (shards an older `append` / `merge` wrote, which decoded the
/// dictionary to its value type first; shards the in-place writers rewrote
/// before they stopped doing the same; and shards appended from a source that
/// held the column plain). For the plain case it
/// builds a local first-seen dedup: `local_codes[row]` is the local index of
/// that row's value (or `-1` when the row is null, keyed on *validity* — a
/// genuine empty-string category is distinct from a null), and `local_values`
/// lists the distinct values in first-seen order.
///
/// Generic over the value class (string / integer / float) so numeric
/// categoricals reconcile exactly as strings do — matching
/// `reconcile_dictionary_representations` on the read side. The downstream
/// `remap!` then folds the result into the cross-shard `CatAccum` identically
/// for both representations (a value-class mismatch vs the accumulator — e.g. a
/// plain `Int64` shard under a `Dictionary(_, Utf8)` column — is rejected by the
/// `remap!` match's catch-all arm; §3.2's validator relax rejects it earlier).
pub(super) fn local_categorical_view(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<i32>, CatValues), ConvertError> {
    if matches!(array.data_type(), DataType::Dictionary(_, _)) {
        return Ok((
            dict_codes_i32(array, name)?,
            dict_category_values(array, name)?,
        ));
    }

    let n = array.len();
    let mut local_codes = vec![-1i32; n];
    macro_rules! intern_int {
        ($t:ty) => {{
            let a = array.as_primitive::<$t>();
            let mut order: Vec<i64> = Vec::new();
            let mut seen: HashMap<i64, i32> = HashMap::new();
            for i in 0..n {
                if a.is_valid(i) {
                    let v = cat_int_to_i64(a.value(i), name)?;
                    let code = match seen.get(&v) {
                        Some(&c) => c,
                        None => {
                            let c: i32 =
                                order.len().try_into().map_err(|_| cardinality_err(name))?;
                            seen.insert(v, c);
                            order.push(v);
                            c
                        }
                    };
                    local_codes[i] = code;
                }
            }
            Ok((local_codes, CatValues::Int(order)))
        }};
    }
    macro_rules! intern_float {
        ($t:ty) => {{
            let a = array.as_primitive::<$t>();
            let mut order: Vec<f64> = Vec::new();
            let mut seen: HashMap<u64, i32> = HashMap::new();
            for i in 0..n {
                if a.is_valid(i) {
                    let v = a.value(i) as f64;
                    let k = v.to_bits();
                    let code = match seen.get(&k) {
                        Some(&c) => c,
                        None => {
                            let c: i32 =
                                order.len().try_into().map_err(|_| cardinality_err(name))?;
                            seen.insert(k, c);
                            order.push(v);
                            c
                        }
                    };
                    local_codes[i] = code;
                }
            }
            Ok((local_codes, CatValues::Float(order)))
        }};
    }
    macro_rules! intern_str {
        ($a:expr) => {{
            let a = $a;
            let mut order: Vec<VarLenUnicode> = Vec::new();
            // Key by `&str` borrowed from `a` (valid for this block) to avoid
            // allocating a `String` per distinct category in this hot path.
            let mut seen: HashMap<&str, i32> = HashMap::new();
            for i in 0..n {
                if a.is_valid(i) {
                    let v = a.value(i);
                    let code = match seen.get(v) {
                        Some(&c) => c,
                        None => {
                            let c: i32 =
                                order.len().try_into().map_err(|_| cardinality_err(name))?;
                            seen.insert(v, c);
                            order.push(vlu(v));
                            c
                        }
                    };
                    local_codes[i] = code;
                }
            }
            Ok((local_codes, CatValues::Str(order)))
        }};
    }
    match array.data_type() {
        DataType::Utf8 => intern_str!(array.as_string::<i32>()),
        DataType::LargeUtf8 => intern_str!(array.as_string::<i64>()),
        DataType::Int8 => intern_int!(Int8Type),
        DataType::Int16 => intern_int!(Int16Type),
        DataType::Int32 => intern_int!(Int32Type),
        DataType::Int64 => intern_int!(Int64Type),
        DataType::UInt8 => intern_int!(UInt8Type),
        DataType::UInt16 => intern_int!(UInt16Type),
        DataType::UInt32 => intern_int!(UInt32Type),
        DataType::UInt64 => intern_int!(UInt64Type),
        DataType::Float32 => intern_float!(Float32Type),
        DataType::Float64 => intern_float!(Float64Type),
        other => Err(ConvertError::Other(format!(
            "column '{name}': categorical column has unsupported plain shard type {other:?}"
        ))),
    }
}

/// Whether the streaming/eager categorical writers can preserve a
/// dictionary with this value type. Mirrors the dtypes [`dict_category_values`]
/// handles (string + every integer / unsigned / float width); anything else
/// (e.g. `Dictionary(_, Boolean)`) takes the warn-and-skip path.
pub(super) fn is_supported_cat_value_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Int8
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

/// Cross-shard global categorical accumulator for the streaming exporter.
/// One variant per category value class (matching [`CatValues`]); the `dict`
/// maps a category to its global code and `order` preserves insertion order
/// for the final `categories` dataset. Floats are deduped by bit pattern
/// (`f64::to_bits`) — categorical category values are exact (cluster labels,
/// dose levels), and pandas does not emit `NaN` categories.
pub(super) enum CatAccum {
    Str {
        dict: HashMap<String, i32>,
        order: Vec<String>,
    },
    Int {
        dict: HashMap<i64, i32>,
        order: Vec<i64>,
    },
    Float {
        dict: HashMap<u64, i32>,
        order: Vec<f64>,
    },
}

impl CatAccum {
    /// Pick the accumulator variant for a dictionary value type. The caller
    /// only reaches this for types accepted by [`is_supported_cat_value_type`];
    /// non-numeric, non-string types fall back to `Str` (unreachable in
    /// practice).
    pub(super) fn new(value_type: &DataType) -> Self {
        match value_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => CatAccum::Int {
                dict: HashMap::new(),
                order: Vec::new(),
            },
            DataType::Float32 | DataType::Float64 => CatAccum::Float {
                dict: HashMap::new(),
                order: Vec::new(),
            },
            _ => CatAccum::Str {
                dict: HashMap::new(),
                order: Vec::new(),
            },
        }
    }
}

/// A shard presents a categorical column in a different value class than the
/// column was declared with. Shared by the pre-scan and the write pass so the
/// two report the same corruption identically.
pub(super) fn cat_class_mismatch_err(name: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': categorical value type changed across shards"
    ))
}

/// Two categorical value types belong to the same value *class* for export
/// reconciliation: `Utf8`/`LargeUtf8` are interchangeable; every other type
/// must match exactly. Used to reject a dictionary whose value class differs
/// across shards (corruption) while allowing the harmless narrow/wide string
/// difference the per-shard batches legitimately carry.
///
/// Numeric widths are required to match **exactly** here (e.g. a plain `Int32`
/// shard under a `Dictionary(_, Int64)` column is rejected), which is
/// intentionally stricter than the read-side `reconcile_dictionary_representations`,
/// where `arrow::compute::cast` would coerce integer widths. No real writer
/// produces a width-mismatched layout — every writer carries the dictionary
/// through, and the decode `append` / `merge` used to apply preserved the
/// exact value type `V` — so the only way to hit the difference is a
/// hand-crafted third-party file, where failing loudly on export is preferable
/// to a silent width coercion.
pub(super) fn cat_value_class_eq(a: &DataType, b: &DataType) -> bool {
    fn is_string_like(t: &DataType) -> bool {
        matches!(t, DataType::Utf8 | DataType::LargeUtf8)
    }
    a == b || (is_string_like(a) && is_string_like(b))
}
