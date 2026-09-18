//! Arrow IPC serialization compatibility shims.
//!
//! Arrow IPC `Utf8` and `Binary` columns use 32-bit signed offsets, capping
//! any single string/binary buffer at `2^31 − 1` ≈ 2.15 GB. SCX files with
//! many millions of rows and string-heavy obs metadata overflow this limit
//! during `arrow::ipc::writer::FileWriter::write()` with errors like
//! `Offset overflow error: 2162454931`.
//!
//! [`upcast_to_large_types`] widens `Utf8 → LargeUtf8` and `Binary →
//! LargeBinary` (which use 64-bit offsets) before IPC serialization. It
//! always widens eligible columns, including ones already in memory as
//! `LargeUtf8` (a no-op).
//!
//! [`downcast_large_types`] is the inverse — applied after deserialization
//! so callers see the canonical narrow types in memory **whenever the data
//! actually fits**. The downcast is *opportunistic*: if a `LargeUtf8` /
//! `LargeBinary` column's last offset exceeds `i32::MAX` (i.e. the
//! offsets cannot be re-expressed as `i32`), the column is left wide and
//! surfaces as `LargeUtf8` / `LargeBinary` to callers. An eager downcast
//! would call `arrow::compute::cast`, which errors with
//! `LargeUtf8 array too large to cast to Utf8 array` on overflow — the
//! exact failure mode of the >2 GB obs files this module exists to
//! support. Predicate evaluators (`scx-engine`) already match on both
//! narrow and wide variants; downstream string-consumers that cannot
//! tolerate `LargeUtf8` are documented per-call.

use arrow::array::{Array, ArrayRef, AsArray, LargeBinaryArray, LargeStringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::error::{Result, ScxError};

/// Upcast `Utf8 → LargeUtf8` and `Binary → LargeBinary` (including
/// `Dictionary(_, Utf8|Binary)` value types) so Arrow IPC uses 64-bit
/// offsets. No-op when no narrow string/binary columns are present.
pub fn upcast_to_large_types(batch: &RecordBatch) -> Result<RecordBatch> {
    convert(batch, true)
}

/// Opportunistically downcast `LargeUtf8 → Utf8` and `LargeBinary →
/// Binary` (including `Dictionary(_, LargeUtf8|LargeBinary)` value
/// types) after IPC deserialization. Columns whose offsets exceed
/// `i32::MAX` are left wide so the >2 GB obs case still reads. No-op
/// when no wide columns are present (or all wide columns overflow).
pub fn downcast_large_types(batch: &RecordBatch) -> Result<RecordBatch> {
    convert(batch, false)
}

/// Schema-only counterpart to [`downcast_large_types`].
///
/// Eagerly rewrites `LargeUtf8 → Utf8` and `LargeBinary → Binary`
/// (including `Dictionary(_, LargeUtf8|LargeBinary)` value types) at
/// the schema level. Unlike the batch-level helper, this version is
/// **unconditional**: there are no column offsets to inspect, so it
/// always narrows. Preserves field-level and schema-level metadata.
///
/// Used by callers (e.g. `scx-cloud::CloudReader::read_obs_schema`)
/// that need a schema for predicate parsing without paying to decode
/// the full IPC batch. The local `ScxReader::read_obs_schema` already
/// achieves the same effect by routing through its fast/slow path
/// (see `scx-format-io/src/reader/metadata.rs`).
pub fn downcast_large_types_schema(schema: &Schema) -> Schema {
    let new_fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            let new_dt = match f.data_type() {
                DataType::LargeUtf8 => DataType::Utf8,
                DataType::LargeBinary => DataType::Binary,
                DataType::Dictionary(k, v) => match v.as_ref() {
                    DataType::LargeUtf8 => {
                        DataType::Dictionary(k.clone(), Box::new(DataType::Utf8))
                    }
                    DataType::LargeBinary => {
                        DataType::Dictionary(k.clone(), Box::new(DataType::Binary))
                    }
                    _ => f.data_type().clone(),
                },
                other => other.clone(),
            };
            Field::new(f.name(), new_dt, f.is_nullable()).with_metadata(f.metadata().clone())
        })
        .collect();
    Schema::new(new_fields).with_metadata(schema.metadata().clone())
}

/// Encode `col` as `Dictionary(Int32, value_type)`.
///
/// `arrow::compute::cast` does this for most value types, but its dictionary
/// *packing* has no `Boolean` support and fails with *"Unsupported output
/// type for dictionary packing: Boolean"*. That is the same limitation
/// [`widen_dictionary_keys`] sidesteps by casting keys. It was reachable by an
/// ordinary `append`, which decoded the rows it added: the base shards kept
/// their `Dictionary(_, Boolean)` while the appended obs shard landed as plain
/// `Boolean`, so [`reconcile_dictionary_representations`] had to encode the
/// plain side — and the whole assembled read (`read_obs`, `to_anndata`,
/// filtered collect) died there. `append` no longer decodes, but the mix is
/// still reachable on a file an older scx version grew and by appending a
/// plain-`Boolean` source onto a dictionary-encoded base.
///
/// A `Boolean` column has at most three states, so it is packed by hand,
/// nulls preserved as null keys. Packing by hand rather than degrading the
/// column to a plain `bool` array keeps it a categorical, so an `append`
/// does not silently change a column's pandas dtype.
///
/// Only the values **actually present** are installed, in first-occurrence
/// order. Unconditionally emitting `[false, true]` is the obvious shortcut
/// and is wrong: it *invents* a category. A column whose declared vocabulary
/// is `[True]` came back as `CategoricalDtype(categories=[True, False])`
/// after an append — visible in `.cat.categories`, in dtype equality, in
/// `groupby(observed=False)`, and in any categorical encoder. The dictionary
/// shards on the other side of the reconcile contribute their own declared
/// categories, and the later concat + dedup unions the two, so nothing is
/// lost by packing only what this shard holds.
fn encode_to_dictionary(col: &ArrayRef, value_type: &DataType) -> Result<ArrayRef> {
    use std::sync::Arc;

    if !matches!(value_type, DataType::Boolean) {
        return Ok(arrow::compute::cast(
            col,
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(value_type.clone())),
        )?);
    }

    // Decode to plain Boolean first: `col` may itself be a dictionary with a
    // key width other than Int32, which is the other half of the mixed case.
    let plain = if matches!(col.data_type(), DataType::Dictionary(_, _)) {
        arrow::compute::cast(col, &DataType::Boolean)?
    } else {
        col.clone()
    };
    let plain = plain
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .ok_or_else(|| {
            ScxError::InvalidCatalog(
                "column declared Boolean but failed downcast while encoding a dictionary".into(),
            )
        })?;

    // First-occurrence order, matching the dedup helpers in `reader/metadata.rs`.
    let mut present: Vec<bool> = Vec::with_capacity(2);
    for i in 0..plain.len() {
        if !plain.is_null(i) {
            let v = plain.value(i);
            if !present.contains(&v) {
                present.push(v);
                if present.len() == 2 {
                    break;
                }
            }
        }
    }
    let keys: arrow::array::Int32Array = (0..plain.len())
        .map(|i| {
            (!plain.is_null(i)).then(|| {
                present
                    .iter()
                    .position(|&p| p == plain.value(i))
                    .expect("every non-null value was interned above") as i32
            })
        })
        .collect();
    let values: ArrayRef = Arc::new(arrow::array::BooleanArray::from(present));
    Ok(Arc::new(arrow::array::DictionaryArray::<
        arrow::datatypes::Int32Type,
    >::try_new(keys, values)?))
}

/// Cast every `Dictionary(K, V)` column's key (index) type to `Int32` so that
/// concatenating per-shard categoricals during sharded-metadata assembly cannot
/// overflow a narrow per-shard key (`Int8`/`Int16`) once the combined vocabulary
/// across shards exceeds that key's range. The value type `V` is preserved;
/// non-dictionary (and already-`Int32`-keyed) columns pass through unchanged.
///
/// Implemented by casting the **keys** array and rebuilding the dictionary
/// over the same values. `Int32` is always wide enough: the combined pre-dedup
/// dictionary length is bounded by the total row count, far below `i32::MAX`.
/// Field- and schema-level metadata (notably the `pandas` index envelope) are
/// preserved.
///
/// This deliberately does **not** go through decode→re-encode
/// (`cast` to `V`, then `cast` to `Dictionary(Int32, V)`). Arrow's dictionary
/// *packing* supports only some value types — `Boolean` is not among them — so
/// re-encoding raised `Unsupported output type for dictionary packing:
/// Boolean` on a `pd.Categorical([True, False])` column. Since this function
/// runs inside [`crate::reader::assemble_sharded_metadata`], that made
/// `read_obs()` fail outright on any **row-sharded** file carrying a boolean
/// categorical, while the same column in an unsharded file read back fine.
/// Casting keys is also cheaper (no `n_obs`-length transient) and preserves
/// the dictionary exactly, including entries no row references.
pub fn widen_dictionary_keys(batch: &RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();
    let needs = schema.fields().iter().any(
        |f| matches!(f.data_type(), DataType::Dictionary(k, _) if k.as_ref() != &DataType::Int32),
    );
    if !needs {
        return Ok(batch.clone());
    }
    let mut new_fields = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        match field.data_type() {
            DataType::Dictionary(k, value_type) if k.as_ref() != &DataType::Int32 => {
                let dict = col.as_any_dictionary_opt().ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "column '{}' declared Dictionary but failed downcast",
                        field.name()
                    ))
                })?;
                let wide_keys = arrow::compute::cast(dict.keys(), &DataType::Int32)?;
                let wide_keys = wide_keys
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| {
                        ScxError::InvalidCatalog(format!(
                            "column '{}': dictionary keys did not cast to Int32",
                            field.name()
                        ))
                    })?
                    .clone();
                let wide_dt = DataType::Dictionary(Box::new(DataType::Int32), value_type.clone());
                new_columns.push(std::sync::Arc::new(arrow::array::DictionaryArray::<
                    arrow::datatypes::Int32Type,
                >::try_new(
                    wide_keys, dict.values().clone()
                )?));
                new_fields.push(
                    Field::new(field.name(), wide_dt, field.is_nullable())
                        .with_metadata(field.metadata().clone()),
                );
            }
            _ => {
                new_columns.push(col.clone());
                new_fields.push(field.as_ref().clone());
            }
        }
    }
    let new_schema = Schema::new(new_fields).with_metadata(schema.metadata().clone());
    Ok(RecordBatch::try_new(
        std::sync::Arc::new(new_schema),
        new_columns,
    )?)
}

/// Reconcile per-shard columns that disagree on `Dictionary`-vs-plain encoding
/// so they can be concatenated.
///
/// Every SCX write door now writes a categorical as a `Dictionary`, but a
/// sharded axis can still carry the column as `Dictionary(_, V)` in some shards
/// and plain `V` in others. Three ways to get there: a file an older `append` /
/// `merge` grew (they decoded categoricals before write), a file whose obs was
/// rewritten in place before those writers stopped casting, and — still live —
/// an `append` whose *source* holds the column plain, since these ops preserve
/// the representation they are handed rather than promoting a plain column. `arrow::compute::concat_batches` requires every batch to share one
/// schema, so it rejects the mix with *"It is not possible to concatenate arrays
/// of different data types (Dictionary(Int32, LargeUtf8), LargeUtf8)"*.
///
/// For each field that is `Dictionary(_, V)` in **any** batch, cast every batch's
/// column for that field to `Dictionary(Int32, V)` (encoding the plain columns;
/// a no-op for columns already in that type). Fields that are dictionary in no
/// batch are left untouched. Must run **after** [`upcast_to_large_types`] +
/// [`widen_dictionary_keys`], so the value type `V` and the `Int32` key already
/// agree across shards and the only residual difference is the dictionary
/// wrapper. The captured dictionary `Field` (name + metadata, e.g. categorical
/// attrs) is applied uniformly so `concat` sees identical schemas.
///
/// Returns the batches unchanged when no field is mixed (homogeneous all-dict or
/// all-plain), so the common single-source-write read path is byte-unaffected.
pub fn reconcile_dictionary_representations(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    if batches.len() < 2 {
        return Ok(batches);
    }
    let schema = batches[0].schema();
    let n_fields = schema.fields().len();

    // For each field, capture a dictionary `Field` if ANY batch carries it as a
    // dictionary, and record whether the representation is mixed across batches.
    let mut dict_field: Vec<Option<Field>> = vec![None; n_fields];
    let mut any_non_dict: Vec<bool> = vec![false; n_fields];
    for batch in &batches {
        // Defensive: this scan indexes the per-field vectors positionally, so a
        // shard whose schema disagrees on field count/order with the first
        // shard would index out of bounds (more fields) or silently mismatch
        // columns. In practice the upstream upcast/widen preserve field
        // identity and `concat_batches` would reject a mismatch anyway, but a
        // malformed / third-party file should fail with a clear error here
        // rather than panic the process.
        let bfields = batch.schema().fields().clone();
        if bfields.len() != n_fields {
            return Err(crate::error::ScxError::InvalidCatalog(format!(
                "sharded metadata shards disagree on field count: expected {n_fields}, \
                 got {} in another shard",
                bfields.len()
            )));
        }
        for (i, field) in bfields.iter().enumerate() {
            if field.name() != schema.field(i).name() {
                return Err(crate::error::ScxError::InvalidCatalog(format!(
                    "sharded metadata shards disagree on column {i}: expected '{}', got '{}'",
                    schema.field(i).name(),
                    field.name()
                )));
            }
            match field.data_type() {
                DataType::Dictionary(_, _) => {
                    if dict_field[i].is_none() {
                        dict_field[i] = Some(field.as_ref().clone());
                    }
                }
                _ => any_non_dict[i] = true,
            }
        }
    }

    // A field needs reconciling iff it is dictionary in some batch and plain in
    // another. If nothing is mixed, every batch is already concat-compatible.
    let mixed: Vec<bool> = (0..n_fields)
        .map(|i| dict_field[i].is_some() && any_non_dict[i])
        .collect();
    if !mixed.iter().any(|&m| m) {
        return Ok(batches);
    }

    batches
        .into_iter()
        .map(|batch| {
            let bschema = batch.schema();
            let mut new_fields = Vec::with_capacity(n_fields);
            let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(n_fields);
            for (i, field) in bschema.fields().iter().enumerate() {
                if mixed[i] {
                    // Target: Int32-keyed dictionary over the existing value type.
                    let target_field = dict_field[i]
                        .as_ref()
                        .expect("mixed field must have a captured dictionary field");
                    let DataType::Dictionary(_, value_type) = target_field.data_type() else {
                        unreachable!("captured field is dictionary by construction");
                    };
                    let target_dt =
                        DataType::Dictionary(Box::new(DataType::Int32), value_type.clone());
                    let col = batch.column(i);
                    let cast_col = if col.data_type() == &target_dt {
                        col.clone()
                    } else {
                        encode_to_dictionary(col, value_type.as_ref())?
                    };
                    new_columns.push(cast_col);
                    // Inherit the captured dictionary field's metadata (categorical
                    // attrs) but keep this batch's nullability for the column.
                    new_fields.push(
                        Field::new(target_field.name(), target_dt, field.is_nullable())
                            .with_metadata(target_field.metadata().clone()),
                    );
                } else {
                    new_columns.push(batch.column(i).clone());
                    new_fields.push(field.as_ref().clone());
                }
            }
            let new_schema = Schema::new(new_fields).with_metadata(bschema.metadata().clone());
            Ok(RecordBatch::try_new(
                std::sync::Arc::new(new_schema),
                new_columns,
            )?)
        })
        .collect()
}

/// True if `col`'s value-offsets buffer can be re-expressed as `i32`
/// (i.e. last offset ≤ `i32::MAX`). Returns `true` for any non-wide
/// type. Used by the downcast path to decide between narrowing and
/// preserving the wide encoding.
fn fits_in_narrow_offsets(col: &ArrayRef) -> bool {
    const LIMIT: i64 = i32::MAX as i64;
    match col.data_type() {
        DataType::LargeUtf8 => col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .and_then(|a| a.value_offsets().last().copied())
            .is_none_or(|n| n <= LIMIT),
        DataType::LargeBinary => col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .and_then(|a| a.value_offsets().last().copied())
            .is_none_or(|n| n <= LIMIT),
        DataType::Dictionary(_, v)
            if matches!(v.as_ref(), DataType::LargeUtf8 | DataType::LargeBinary) =>
        {
            // Dictionary values live as the first child array. Its
            // offsets buffer is buffer 0 (i64 stride for Large* types).
            let data = col.to_data();
            let Some(values) = data.child_data().first() else {
                return true;
            };
            let Some(buf) = values.buffers().first() else {
                return true;
            };
            buf.typed_data::<i64>()
                .last()
                .copied()
                .is_none_or(|n| n <= LIMIT)
        }
        _ => true,
    }
}

fn convert(batch: &RecordBatch, widen: bool) -> Result<RecordBatch> {
    let schema = batch.schema();
    // `target_for` returns the desired datatype iff this column should be
    // converted. On the upcast path that's any eligible narrow type. On
    // the downcast path it's any wide type whose offsets actually fit
    // back in `i32` — wide-but-overflowing columns return `None` and pass
    // through unchanged.
    let target_for = |field_dt: &DataType, col: &ArrayRef| -> Option<DataType> {
        let inner = match field_dt {
            DataType::Dictionary(_, v) => v.as_ref(),
            other => other,
        };
        let new_inner = match (inner, widen) {
            (DataType::Utf8, true) => DataType::LargeUtf8,
            (DataType::Binary, true) => DataType::LargeBinary,
            (DataType::LargeUtf8, false) => DataType::Utf8,
            (DataType::LargeBinary, false) => DataType::Binary,
            _ => return None,
        };
        if !widen && !fits_in_narrow_offsets(col) {
            return None;
        }
        Some(match field_dt {
            DataType::Dictionary(k, _) => DataType::Dictionary(k.clone(), Box::new(new_inner)),
            _ => new_inner,
        })
    };

    if !schema
        .fields()
        .iter()
        .zip(batch.columns())
        .any(|(f, c)| target_for(f.data_type(), c).is_some())
    {
        return Ok(batch.clone());
    }

    // Preserve schema-level metadata (notably the `b"pandas"` key, which
    // tells pyarrow's `to_pandas()` which column is the index) and
    // field-level metadata. Building the new Schema via `Schema::new`
    // alone would silently drop both and break the AnnData round-trip.
    let mut new_fields = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        if let Some(target_dt) = target_for(field.data_type(), col) {
            new_columns.push(arrow::compute::cast(col, &target_dt)?);
            new_fields.push(
                Field::new(field.name(), target_dt, field.is_nullable())
                    .with_metadata(field.metadata().clone()),
            );
        } else {
            new_columns.push(col.clone());
            new_fields.push(field.as_ref().clone());
        }
    }
    let new_schema = Schema::new(new_fields).with_metadata(schema.metadata().clone());
    Ok(RecordBatch::try_new(
        std::sync::Arc::new(new_schema),
        new_columns,
    )?)
}

/// Parse the `index_columns` array from the Arrow IPC schema's
/// `pandas` metadata key. `pyarrow.Table.from_pandas(df)` stamps this
/// with a JSON envelope of the form
/// `{"index_columns": ["gene_symbols", ...], ...}` for string-named
/// indexes (and `__index_level_0__` for unnamed indexes). For
/// `RangeIndex`, pyarrow emits a dict envelope instead of a string;
/// we filter to string entries since `RangeIndex` never names a row.
/// Returns an empty vec when the key is absent, malformed, or holds
/// only dict envelopes.
pub fn pandas_index_columns(schema: &Schema) -> Vec<String> {
    let Some(raw) = schema.metadata().get("pandas") else {
        return Vec::new();
    };
    let parsed: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    parsed
        .get("index_columns")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Field names that carry the frame's index, tolerating a missing envelope.
///
/// [`pandas_index_columns`] reads only the authoritative `pandas` schema
/// metadata. That envelope is absent on two kinds of file: those written by
/// the CLI convert path, and — until the schema-metadata carry fix — any file
/// whose obs was rewritten in place by `append` / `merge` / `modify_metadata` /
/// `attach_external_obs`, all of which rebuilt the schema without it.
///
/// So consumers that need to *identify* the index (rather than merely honour a
/// declared one) fall back to the literal names pyarrow and anndata use.
/// `__index_level_0__` is probed before `_index` so pyarrow's spelling wins on
/// a round-tripped frame that somehow carries both.
///
/// Returns empty when neither is present — a frame with no index column is a
/// real state, and inventing one would be worse than reporting none.
pub fn resolve_index_columns(schema: &Schema) -> Vec<String> {
    let declared = pandas_index_columns(schema);
    if !declared.is_empty() {
        return declared;
    }
    ["__index_level_0__", "_index"]
        .iter()
        .find(|name| schema.field_with_name(name).is_ok())
        .map(|s| vec![s.to_string()])
        .unwrap_or_default()
}

/// Defensive: ensure the schema's `pandas` metadata envelope identifies
/// the index column when the batch carries a literal `__index_level_0__`
/// or `_index` field but no metadata. anndata 0.10+ hard-rejects any
/// DataFrame column called `_index` on `write_h5ad`, so a missing
/// envelope here means downstream `pyarrow.Table.to_pandas()` will leak
/// `_index` into `df.columns` and the user's first `adata.write_h5ad`
/// crashes. The B1 reader-side fix stamps this envelope on every
/// CLI-produced SCX; this helper is the belt-and-braces for any other
/// path (legacy files, third-party writers, future regressions).
///
/// Behaviour:
/// - If `pandas_index_columns` already returns a non-empty list →
///   return `batch.clone()` (no-op).
/// - Else take [`resolve_index_columns`]'s literal-field fallback
///   (`__index_level_0__`, then `_index`).
/// - If neither is present → return `batch.clone()` (no spurious
///   metadata for files that legitimately have no index column).
pub fn ensure_pandas_index_metadata(batch: &RecordBatch) -> RecordBatch {
    if !pandas_index_columns(batch.schema_ref()).is_empty() {
        return batch.clone();
    }
    let schema = batch.schema();
    // Same probe as `resolve_index_columns`, single-sourced: what this stamps
    // and what the export/projection paths resolve must never disagree.
    let Some(idx_name) = resolve_index_columns(schema.as_ref()).into_iter().next() else {
        return batch.clone();
    };
    let mut metadata = schema.metadata().clone();
    metadata.insert(
        "pandas".to_string(),
        serde_json::json!({"index_columns": [idx_name]}).to_string(),
    );
    let new_schema = Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect::<Vec<_>>(),
    )
    .with_metadata(metadata);
    // Safe: same arrays, same field count, same dtypes.
    RecordBatch::try_new(std::sync::Arc::new(new_schema), batch.columns().to_vec())
        .expect("ensure_pandas_index_metadata: schema/columns mismatch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, BinaryArray, DictionaryArray, Float32Array, Int64Array, LargeBinaryArray,
        LargeStringArray, StringArray,
    };
    use arrow::datatypes::Int8Type;
    use std::sync::Arc;

    fn batch_from(fields: Vec<(&str, DataType, ArrayRef)>) -> RecordBatch {
        let schema = Arc::new(Schema::new(
            fields
                .iter()
                .map(|(name, dt, _)| Field::new(*name, dt.clone(), true))
                .collect::<Vec<_>>(),
        ));
        let columns: Vec<ArrayRef> = fields.into_iter().map(|(_, _, arr)| arr).collect();
        RecordBatch::try_new(schema, columns).unwrap()
    }

    #[test]
    fn widen_dictionary_keys_promotes_int8_to_int32_preserving_values() {
        use arrow::datatypes::Int32Type;
        let dict: DictionaryArray<Int8Type> = vec!["a", "b", "a", "c"].into_iter().collect();
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let batch = batch_from(vec![("ct", dict_dt, Arc::new(dict))]);

        let widened = widen_dictionary_keys(&batch).unwrap();
        assert_eq!(
            widened.schema().field(0).data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        );
        let got = widened
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        let values = got.values().as_any().downcast_ref::<StringArray>().unwrap();
        let keys = got.keys();
        let decoded: Vec<&str> = (0..got.len())
            .map(|i| values.value(keys.value(i) as usize))
            .collect();
        assert_eq!(decoded, vec!["a", "b", "a", "c"]);
    }

    /// `pd.Categorical([True, False])` reaches Arrow as
    /// `Dictionary(Int8, Boolean)`. Widening its keys must not depend on
    /// re-encoding the values: arrow's dictionary packing has no `Boolean`
    /// support, so the decode→re-encode round trip raised *"Unsupported
    /// output type for dictionary packing: Boolean"* — and because this
    /// runs inside `assemble_sharded_metadata`, that made `read_obs()`
    /// fail outright on any **row-sharded** file carrying a boolean
    /// categorical. The file wrote without complaint; only reading it back
    /// failed, and only at the scale where obs is sharded.
    #[test]
    fn widen_dictionary_keys_handles_non_packable_value_types() {
        use arrow::array::{Array, BooleanArray};
        use arrow::datatypes::Int32Type;

        let keys = arrow::array::Int8Array::from(vec![Some(0), Some(1), None, Some(0)]);
        let values: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let dict: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap());
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Boolean));
        let batch = batch_from(vec![("flag", dict_dt, dict)]);

        let widened = widen_dictionary_keys(&batch).expect("boolean categorical must widen");
        assert_eq!(
            widened.schema().field(0).data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean))
        );
        let got = widened
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        let values = got
            .values()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let decoded: Vec<Option<bool>> = (0..got.len())
            .map(|i| (!got.keys().is_null(i)).then(|| values.value(got.keys().value(i) as usize)))
            .collect();
        assert_eq!(
            decoded,
            vec![Some(true), Some(false), None, Some(true)],
            "values and nulls must survive the key widening"
        );
    }

    /// Regression: the mixed dictionary/plain shape `append` creates.
    ///
    /// `append` raw-copies the base `Dictionary(_, Boolean)` shards while the
    /// appended obs shard lands as plain `Boolean`, so reconcile has to encode
    /// the plain side — and `arrow::compute::cast` cannot pack a `Boolean`
    /// into a dictionary. Fixing only `widen_dictionary_keys` and the dedup
    /// arm left this third site raising the same error on `read_obs()` after
    /// an ordinary append.
    #[test]
    fn reconcile_handles_a_mixed_boolean_dictionary_and_plain_column() {
        use arrow::array::{Array, BooleanArray};
        use arrow::datatypes::Int32Type;

        let dict_dt = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean));
        let keys = arrow::array::Int32Array::from(vec![Some(0), Some(1), None]);
        let values: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let dict: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap());
        let plain: ArrayRef = Arc::new(BooleanArray::from(vec![Some(false), Some(true)]));

        let out = reconcile_dictionary_representations(vec![
            batch_from(vec![("flag", dict_dt.clone(), dict)]),
            batch_from(vec![("flag", DataType::Boolean, plain)]),
        ])
        .expect("a mixed boolean column must reconcile");

        // Both batches must end up on the same dtype, or `concat` rejects them.
        assert_eq!(out[0].schema().field(0).data_type(), &dict_dt);
        assert_eq!(out[1].schema().field(0).data_type(), &dict_dt);

        // And the values must survive on the encoded side.
        let decoded = arrow::compute::cast(out[1].column(0), &DataType::Boolean).unwrap();
        let decoded = decoded.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            (0..decoded.len())
                .map(|i| (!decoded.is_null(i)).then(|| decoded.value(i)))
                .collect::<Vec<_>>(),
            vec![Some(false), Some(true)]
        );

        // The pair must actually concatenate — the failure this guards is a
        // read that dies while assembling a sharded axis — and the merged
        // column must decode to the right values. "concat succeeded" alone
        // would let a key-remapping regression hide: the two sides intern
        // their values in different orders (`[true, false]` vs `[false]`),
        // so a naive concat that kept both key spaces would silently invert
        // the appended rows.
        let schema = out[0].schema();
        let merged =
            arrow::compute::concat_batches(&schema, &out).expect("reconciled batches must concat");
        let decoded = arrow::compute::cast(merged.column(0), &DataType::Boolean).unwrap();
        let decoded = decoded.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            (0..decoded.len())
                .map(|i| (!decoded.is_null(i)).then(|| decoded.value(i)))
                .collect::<Vec<_>>(),
            vec![Some(true), Some(false), None, Some(false), Some(true)],
            "merged column must preserve every row's value across both key spaces"
        );
    }

    /// The hand-packed encoder must not **invent** a category. Emitting
    /// `[false, true]` unconditionally is the obvious shortcut and turns a
    /// column whose declared vocabulary is `[True]` into `[True, False]` —
    /// visible in `.cat.categories`, dtype equality, and
    /// `groupby(observed=False)`.
    #[test]
    fn boolean_encoding_installs_only_the_values_present() {
        use arrow::array::BooleanArray;
        use arrow::datatypes::Int32Type;

        let dict_dt = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean));
        let keys = arrow::array::Int32Array::from(vec![Some(0), Some(0)]);
        let values: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
        let dict: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap());
        // The plain side holds only `true` — so must `false` stay out of the
        // vocabulary.
        let plain: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), Some(true)]));

        let out = reconcile_dictionary_representations(vec![
            batch_from(vec![("flag", dict_dt, dict)]),
            batch_from(vec![("flag", DataType::Boolean, plain)]),
        ])
        .unwrap();

        let encoded = out[1].column(0).as_any_dictionary_opt().unwrap();
        let vocab = encoded
            .values()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(
            (0..vocab.len()).map(|i| vocab.value(i)).collect::<Vec<_>>(),
            vec![true],
            "only the values actually present may be installed"
        );
    }

    /// An all-null Boolean column has no values to intern at all. The
    /// encoder must still produce a well-formed dictionary rather than
    /// panicking or inventing a vocabulary for rows that have none.
    #[test]
    fn boolean_encoding_handles_an_all_null_column() {
        use arrow::array::BooleanArray;
        use arrow::datatypes::Int32Type;

        let dict_dt = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean));
        let keys = arrow::array::Int32Array::from(vec![Some(0)]);
        let values: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
        let dict: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap());
        let plain: ArrayRef = Arc::new(BooleanArray::from(vec![None, None] as Vec<Option<bool>>));

        let out = reconcile_dictionary_representations(vec![
            batch_from(vec![("flag", dict_dt.clone(), dict)]),
            batch_from(vec![("flag", DataType::Boolean, plain)]),
        ])
        .expect("an all-null boolean column must reconcile");
        let schema = out[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &out).unwrap();
        let decoded = arrow::compute::cast(merged.column(0), &DataType::Boolean).unwrap();
        let decoded = decoded.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            (0..decoded.len())
                .map(|i| (!decoded.is_null(i)).then(|| decoded.value(i)))
                .collect::<Vec<_>>(),
            vec![Some(true), None, None]
        );
    }

    #[test]
    fn widen_dictionary_keys_passes_through_non_dictionary() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["x", "y"]));
        let batch = batch_from(vec![("s", DataType::Utf8, arr)]);
        let out = widen_dictionary_keys(&batch).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Utf8);
    }

    #[test]
    fn reconcile_is_noop_on_homogeneous_batches() {
        // All-plain: untouched.
        let a: ArrayRef = Arc::new(StringArray::from(vec!["x", "y"]));
        let b: ArrayRef = Arc::new(StringArray::from(vec!["z"]));
        let plain = vec![
            batch_from(vec![("s", DataType::Utf8, a)]),
            batch_from(vec![("s", DataType::Utf8, b)]),
        ];
        let out = reconcile_dictionary_representations(plain).unwrap();
        assert_eq!(out[0].schema().field(0).data_type(), &DataType::Utf8);
        assert_eq!(out[1].schema().field(0).data_type(), &DataType::Utf8);

        // All-dictionary: untouched (still Dictionary, same key width).
        let d0: DictionaryArray<Int8Type> = vec!["a", "b"].into_iter().collect();
        let d1: DictionaryArray<Int8Type> = vec!["c"].into_iter().collect();
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let dicts = vec![
            batch_from(vec![("ct", dict_dt.clone(), Arc::new(d0))]),
            batch_from(vec![("ct", dict_dt.clone(), Arc::new(d1))]),
        ];
        let out = reconcile_dictionary_representations(dicts).unwrap();
        assert_eq!(out[0].schema().field(0).data_type(), &dict_dt);
        assert_eq!(out[1].schema().field(0).data_type(), &dict_dt);
    }

    #[test]
    fn reconcile_encodes_plain_shard_to_dictionary_when_mixed() {
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let d0: DictionaryArray<Int8Type> = vec!["a", "b"].into_iter().collect();
        let plain: ArrayRef = Arc::new(StringArray::from(vec!["c", "d"]));
        let mixed = vec![
            batch_from(vec![("ct", dict_dt, Arc::new(d0))]),
            batch_from(vec![("ct", DataType::Utf8, plain)]),
        ];

        let out = reconcile_dictionary_representations(mixed).unwrap();
        let target = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        // Both batches now share one Dictionary schema, so concat succeeds.
        assert_eq!(out[0].schema().field(0).data_type(), &target);
        assert_eq!(out[1].schema().field(0).data_type(), &target);
        let merged = arrow::compute::concat_batches(&out[0].schema(), out.iter()).unwrap();
        assert_eq!(merged.num_rows(), 4);
    }

    #[test]
    fn reconcile_errors_on_mismatched_field_count() {
        // A later shard with an extra column would index the per-field vectors
        // out of bounds; the guard must return a clear error, not panic.
        let a: ArrayRef = Arc::new(StringArray::from(vec!["x"]));
        let b0: ArrayRef = Arc::new(StringArray::from(vec!["y"]));
        let b1: ArrayRef = Arc::new(StringArray::from(vec!["z"]));
        let batches = vec![
            batch_from(vec![("s", DataType::Utf8, a)]),
            batch_from(vec![
                ("s", DataType::Utf8, b0),
                ("extra", DataType::Utf8, b1),
            ]),
        ];
        let err = reconcile_dictionary_representations(batches).unwrap_err();
        assert!(
            err.to_string().contains("field count"),
            "expected a field-count mismatch error, got: {err}"
        );
    }

    #[test]
    fn reconcile_errors_on_mismatched_field_name() {
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let d0: DictionaryArray<Int8Type> = vec!["a"].into_iter().collect();
        let plain: ArrayRef = Arc::new(StringArray::from(vec!["b"]));
        // Same field count, mismatched name at index 0 — would silently graft the
        // dict field onto the wrong column without the name guard.
        let batches = vec![
            batch_from(vec![("ct", dict_dt, Arc::new(d0))]),
            batch_from(vec![("other", DataType::Utf8, plain)]),
        ];
        let err = reconcile_dictionary_representations(batches).unwrap_err();
        assert!(
            err.to_string().contains("column 0"),
            "expected a field-name mismatch error, got: {err}"
        );
    }

    #[test]
    fn utf8_round_trip_preserves_schema_and_values() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"]));
        let batch = batch_from(vec![("cell_id", DataType::Utf8, arr.clone())]);

        let upcast = upcast_to_large_types(&batch).unwrap();
        assert_eq!(upcast.schema().field(0).data_type(), &DataType::LargeUtf8);

        let downcast = downcast_large_types(&upcast).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::Utf8);
        let got = downcast
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(got.value(0), "alpha");
        assert_eq!(got.value(1), "beta");
        assert_eq!(got.value(2), "gamma");
    }

    #[test]
    fn binary_round_trip_preserves_schema_and_values() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            b"\x01\x02".as_ref(),
            b"\x03".as_ref(),
        ]));
        let batch = batch_from(vec![("blob", DataType::Binary, arr)]);

        let upcast = upcast_to_large_types(&batch).unwrap();
        assert_eq!(upcast.schema().field(0).data_type(), &DataType::LargeBinary);

        let downcast = downcast_large_types(&upcast).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::Binary);
        let got = downcast
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(got.value(0), b"\x01\x02");
        assert_eq!(got.value(1), b"\x03");
    }

    #[test]
    fn dictionary_int8_utf8_round_trip_preserves_index_type() {
        let dict: DictionaryArray<Int8Type> = vec!["a", "b", "a", "c"].into_iter().collect();
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let batch = batch_from(vec![("cluster", dict_dt.clone(), Arc::new(dict))]);

        let upcast = upcast_to_large_types(&batch).unwrap();
        let upcast_dt =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::LargeUtf8));
        assert_eq!(upcast.schema().field(0).data_type(), &upcast_dt);

        let downcast = downcast_large_types(&upcast).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &dict_dt);
        let got = downcast
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .unwrap();
        let values = got.values().as_any().downcast_ref::<StringArray>().unwrap();
        let keys = got.keys();
        let collected: Vec<&str> = (0..got.len())
            .map(|i| values.value(keys.value(i) as usize))
            .collect();
        assert_eq!(collected, vec!["a", "b", "a", "c"]);
    }

    #[test]
    fn no_op_when_no_string_or_binary_columns() {
        let f32_arr: ArrayRef = Arc::new(Float32Array::from(vec![1.0_f32, 2.0, 3.0]));
        let i64_arr: ArrayRef = Arc::new(Int64Array::from(vec![10_i64, 20, 30]));
        let batch = batch_from(vec![
            ("x", DataType::Float32, f32_arr),
            ("n", DataType::Int64, i64_arr),
        ]);

        let upcast = upcast_to_large_types(&batch).unwrap();
        assert_eq!(upcast.schema(), batch.schema());
        // Original arrays should be reused verbatim.
        assert!(Arc::ptr_eq(
            &(upcast.column(0).clone() as ArrayRef),
            &(batch.column(0).clone() as ArrayRef)
        ));

        let downcast = downcast_large_types(&batch).unwrap();
        assert_eq!(downcast.schema(), batch.schema());
    }

    #[test]
    fn empty_batch_round_trips() {
        let arr: ArrayRef = Arc::new(StringArray::from(Vec::<&str>::new()));
        let batch = batch_from(vec![("cell_id", DataType::Utf8, arr)]);
        assert_eq!(batch.num_rows(), 0);

        let upcast = upcast_to_large_types(&batch).unwrap();
        assert_eq!(upcast.schema().field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(upcast.num_rows(), 0);

        let downcast = downcast_large_types(&upcast).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::Utf8);
        assert_eq!(downcast.num_rows(), 0);
    }

    #[test]
    fn mixed_columns_only_cast_eligible_ones() {
        let s_arr: ArrayRef = Arc::new(StringArray::from(vec!["x", "y"]));
        let f_arr: ArrayRef = Arc::new(Float32Array::from(vec![1.0_f32, 2.0]));
        let batch = batch_from(vec![
            ("name", DataType::Utf8, s_arr),
            ("score", DataType::Float32, f_arr.clone()),
        ]);

        let upcast = upcast_to_large_types(&batch).unwrap();
        assert_eq!(upcast.schema().field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(upcast.schema().field(1).data_type(), &DataType::Float32);
        // Float column is not re-allocated.
        assert!(Arc::ptr_eq(
            &(upcast.column(1).clone() as ArrayRef),
            &(batch.column(1).clone() as ArrayRef)
        ));
    }

    #[test]
    fn upcast_then_downcast_already_narrow_is_noop() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let batch = batch_from(vec![("cell_id", DataType::Utf8, arr)]);
        let downcast = downcast_large_types(&batch).unwrap();
        // No LargeUtf8 columns -> short-circuit returns clone.
        assert_eq!(downcast.schema(), batch.schema());
    }

    #[test]
    fn large_utf8_input_downcasts_to_utf8() {
        let arr: ArrayRef = Arc::new(LargeStringArray::from(vec!["foo", "bar"]));
        let batch = batch_from(vec![("cell_id", DataType::LargeUtf8, arr)]);

        let downcast = downcast_large_types(&batch).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::Utf8);
        let got = downcast
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(got.value(0), "foo");
        assert_eq!(got.value(1), "bar");
    }

    #[test]
    fn schema_and_field_metadata_are_preserved() {
        use std::collections::HashMap;

        // Schema metadata: pandas integration relies on the `b"pandas"`
        // key (which encodes index column hints). Field metadata is
        // sometimes set for var_names symbol mapping, etc. Both must
        // survive a round-trip or the AnnData / pandas pipeline breaks.
        let mut schema_md = HashMap::new();
        schema_md.insert(
            "pandas".to_string(),
            r#"{"index_columns":["cell_id"]}"#.to_string(),
        );
        schema_md.insert("custom_top".to_string(), "value_top".to_string());

        let mut field_md = HashMap::new();
        field_md.insert("symbol_column".to_string(), "ENSG".to_string());

        let field = Field::new("cell_id", DataType::Utf8, false).with_metadata(field_md.clone());
        let schema = Arc::new(Schema::new(vec![field]).with_metadata(schema_md.clone()));
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let upcast = upcast_to_large_types(&batch).unwrap();
        assert_eq!(upcast.schema().metadata(), &schema_md);
        assert_eq!(upcast.schema().field(0).metadata(), &field_md);
        assert_eq!(upcast.schema().field(0).data_type(), &DataType::LargeUtf8);

        let downcast = downcast_large_types(&upcast).unwrap();
        assert_eq!(downcast.schema().metadata(), &schema_md);
        assert_eq!(downcast.schema().field(0).metadata(), &field_md);
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::Utf8);
    }

    // -------------------------------------------------------------
    // resolve_index_columns: envelope first, literal field as fallback
    // -------------------------------------------------------------

    #[test]
    fn resolve_index_columns_prefers_the_declared_envelope() {
        use std::collections::HashMap;
        // A frame whose envelope names `barcode` while a literal
        // `__index_level_0__` field also exists. The declaration wins: it is
        // authoritative, and guessing over it would override a caller who
        // deliberately named a different index.
        let mut md = HashMap::new();
        md.insert(
            "pandas".to_string(),
            r#"{"index_columns":["barcode"]}"#.to_string(),
        );
        let a1: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let a2: ArrayRef = Arc::new(StringArray::from(vec!["c", "d"]));
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("barcode", DataType::Utf8, true),
                Field::new("__index_level_0__", DataType::Utf8, true),
            ])
            .with_metadata(md),
        );
        let batch = RecordBatch::try_new(schema, vec![a1, a2]).unwrap();
        assert_eq!(
            resolve_index_columns(batch.schema_ref()),
            vec!["barcode".to_string()]
        );
    }

    /// The case that matters for already-written files: no envelope (an obs
    /// rewritten in place before the schema-metadata carry fix), and the index
    /// field is NOT field 0 — pyarrow puts it last. Falling back to field 0
    /// here is what renamed every exported cell to its cell type.
    #[test]
    fn resolve_index_columns_finds_the_literal_field_without_an_envelope() {
        let a1: ArrayRef = Arc::new(StringArray::from(vec!["T", "B"]));
        let a2: ArrayRef = Arc::new(StringArray::from(vec!["c0", "c1"]));
        let batch = batch_from(vec![
            ("cell_type", DataType::Utf8, a1),
            ("__index_level_0__", DataType::Utf8, a2),
        ]);
        assert!(pandas_index_columns(batch.schema_ref()).is_empty());
        assert_eq!(
            resolve_index_columns(batch.schema_ref()),
            vec!["__index_level_0__".to_string()]
        );
    }

    #[test]
    fn resolve_index_columns_accepts_the_anndata_spelling() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let batch = batch_from(vec![("_index", DataType::Utf8, arr)]);
        assert_eq!(
            resolve_index_columns(batch.schema_ref()),
            vec!["_index".to_string()]
        );
    }

    #[test]
    fn resolve_index_columns_prefers_pyarrows_spelling_over_anndatas() {
        let a1: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let a2: ArrayRef = Arc::new(StringArray::from(vec!["c", "d"]));
        let batch = batch_from(vec![
            ("_index", DataType::Utf8, a1),
            ("__index_level_0__", DataType::Utf8, a2),
        ]);
        assert_eq!(
            resolve_index_columns(batch.schema_ref()),
            vec!["__index_level_0__".to_string()]
        );
    }

    /// A frame with genuinely no index column must report none. Inventing one
    /// would be worse than reporting nothing: the export's field-0 fallback is
    /// correct for exactly this shape (CLI-converted obs), and a fabricated
    /// answer here would override it.
    #[test]
    fn resolve_index_columns_reports_none_when_there_is_no_index() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let batch = batch_from(vec![("cell_id", DataType::Utf8, arr)]);
        assert!(resolve_index_columns(batch.schema_ref()).is_empty());
    }

    // -------------------------------------------------------------
    // B2-2026-05-20: ensure_pandas_index_metadata defensive helper
    // -------------------------------------------------------------

    #[test]
    fn ensure_pandas_index_metadata_noop_when_already_set() {
        use std::collections::HashMap;
        let mut schema_md = HashMap::new();
        schema_md.insert(
            "pandas".to_string(),
            r#"{"index_columns":["__index_level_0__"]}"#.to_string(),
        );
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let schema = Arc::new(
            Schema::new(vec![Field::new("__index_level_0__", DataType::Utf8, true)])
                .with_metadata(schema_md.clone()),
        );
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = ensure_pandas_index_metadata(&batch);
        assert_eq!(out.schema().metadata(), &schema_md);
        assert_eq!(
            pandas_index_columns(out.schema_ref()),
            vec!["__index_level_0__".to_string()]
        );
    }

    #[test]
    fn ensure_pandas_index_metadata_promotes_double_underscore() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let batch = batch_from(vec![("__index_level_0__", DataType::Utf8, arr)]);
        assert!(pandas_index_columns(batch.schema_ref()).is_empty());
        let out = ensure_pandas_index_metadata(&batch);
        assert_eq!(
            pandas_index_columns(out.schema_ref()),
            vec!["__index_level_0__".to_string()]
        );
    }

    #[test]
    fn ensure_pandas_index_metadata_promotes_single_underscore() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let batch = batch_from(vec![("_index", DataType::Utf8, arr)]);
        assert!(pandas_index_columns(batch.schema_ref()).is_empty());
        let out = ensure_pandas_index_metadata(&batch);
        assert_eq!(
            pandas_index_columns(out.schema_ref()),
            vec!["_index".to_string()]
        );
    }

    #[test]
    fn ensure_pandas_index_metadata_double_underscore_wins_over_single() {
        // Both `__index_level_0__` and `_index` are present (pathological
        // but legal). pyarrow's canonical name wins.
        let a1: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let a2: ArrayRef = Arc::new(StringArray::from(vec!["c", "d"]));
        let batch = batch_from(vec![
            ("_index", DataType::Utf8, a1),
            ("__index_level_0__", DataType::Utf8, a2),
        ]);
        let out = ensure_pandas_index_metadata(&batch);
        assert_eq!(
            pandas_index_columns(out.schema_ref()),
            vec!["__index_level_0__".to_string()]
        );
    }

    #[test]
    fn ensure_pandas_index_metadata_noop_when_neither_present() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let batch = batch_from(vec![("cell_id", DataType::Utf8, arr)]);
        let out = ensure_pandas_index_metadata(&batch);
        assert!(out.schema().metadata().get("pandas").is_none());
        assert!(pandas_index_columns(out.schema_ref()).is_empty());
    }

    #[test]
    fn large_binary_input_downcasts_to_binary() {
        let arr: ArrayRef = Arc::new(LargeBinaryArray::from(vec![
            b"\xaa".as_ref(),
            b"\xbb\xcc".as_ref(),
        ]));
        let batch = batch_from(vec![("blob", DataType::LargeBinary, arr)]);

        let downcast = downcast_large_types(&batch).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::Binary);
        let got = downcast
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(got.value(0), b"\xaa");
        assert_eq!(got.value(1), b"\xbb\xcc");
    }

    // ---------------------------------------------------------------------
    // Opportunistic-downcast tests
    //
    // The actual >2 GB obs case is exercised end-to-end by an
    // `#[ignore]`-gated integration test in `scx-format/tests/`. These
    // unit tests fake a too-wide column by building `ArrayData` via
    // `build_unchecked` with an offsets buffer whose last entry exceeds
    // `i32::MAX`, sidestepping the multi-GB values-buffer allocation
    // that a "real" overflowing array would require.
    // ---------------------------------------------------------------------

    /// Construct an `ArrayRef` of dtype `dt` with the given `offsets` (i64)
    /// without allocating a full-size values buffer. Bypasses Arrow's
    /// usual "offsets ≤ values.len()" invariant via `build_unchecked` —
    /// safe here because `fits_in_narrow_offsets` and the no-op short-
    /// circuit only touch the offsets buffer.
    fn fake_wide_array(dt: DataType, offsets: &[i64]) -> ArrayRef {
        use arrow::array::ArrayData;
        use arrow::buffer::Buffer;

        let offsets_buf = Buffer::from_slice_ref(offsets);
        // Length of the array is `offsets.len() - 1`. Values buffer is
        // intentionally empty — we never read through it.
        let values_buf = Buffer::from_vec::<u8>(Vec::new());
        let data = unsafe {
            ArrayData::builder(dt.clone())
                .len(offsets.len().saturating_sub(1))
                .add_buffer(offsets_buf)
                .add_buffer(values_buf)
                .build_unchecked()
        };
        match dt {
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(data)),
            DataType::LargeBinary => Arc::new(LargeBinaryArray::from(data)),
            other => unreachable!("fake_wide_array: unsupported dtype {other:?}"),
        }
    }

    #[test]
    fn downcast_preserves_largeutf8_when_offsets_overflow_i32() {
        // Offsets [0, i32::MAX + 1] — last offset cannot be re-expressed
        // as `i32`, so opportunistic downcast must keep the column wide.
        let overflow = i32::MAX as i64 + 1;
        let arr = fake_wide_array(DataType::LargeUtf8, &[0, overflow]);
        let batch = batch_from(vec![("cell_id", DataType::LargeUtf8, arr)]);

        let downcast = downcast_large_types(&batch).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::LargeUtf8);
    }

    #[test]
    fn downcast_preserves_largebinary_when_offsets_overflow_i32() {
        let overflow = i32::MAX as i64 + 1;
        let arr = fake_wide_array(DataType::LargeBinary, &[0, overflow]);
        let batch = batch_from(vec![("blob", DataType::LargeBinary, arr)]);

        let downcast = downcast_large_types(&batch).unwrap();
        assert_eq!(
            downcast.schema().field(0).data_type(),
            &DataType::LargeBinary
        );
    }

    #[test]
    fn downcast_preserves_dictionary_largeutf8_values_when_overflow() {
        // Build a `Dictionary(Int8, LargeUtf8)` whose values dictionary's
        // last offset overflows `i32::MAX`. Opportunistic path keeps it
        // wide.
        use arrow::array::ArrayData;
        use arrow::buffer::Buffer;

        let overflow = i32::MAX as i64 + 1;
        // Inner LargeUtf8 values dict: a single string slot whose last
        // offset claims to be > i32::MAX.
        let values_offsets = Buffer::from_slice_ref([0_i64, overflow]);
        let values_data = unsafe {
            ArrayData::builder(DataType::LargeUtf8)
                .len(1)
                .add_buffer(values_offsets)
                .add_buffer(Buffer::from_vec::<u8>(Vec::new()))
                .build_unchecked()
        };

        // Keys: `[0, 0]` pointing to the single dict entry.
        let keys_buf = Buffer::from_slice_ref([0_i8, 0]);
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::LargeUtf8));
        let dict_data = unsafe {
            ArrayData::builder(dict_dt.clone())
                .len(2)
                .add_buffer(keys_buf)
                .add_child_data(values_data)
                .build_unchecked()
        };
        let arr: ArrayRef =
            Arc::new(arrow::array::DictionaryArray::<arrow::datatypes::Int8Type>::from(dict_data));
        let batch = batch_from(vec![("cluster", dict_dt.clone(), arr)]);

        let downcast = downcast_large_types(&batch).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &dict_dt);
    }

    #[test]
    fn downcast_at_offset_boundary_stays_narrow() {
        // Last offset == i32::MAX is the largest value that can be
        // re-expressed as `i32` — opportunistic path should still
        // downcast.
        let arr = fake_wide_array(DataType::LargeUtf8, &[0, i32::MAX as i64]);
        let batch = batch_from(vec![("cell_id", DataType::LargeUtf8, arr)]);

        // We can't materialize through `arrow::compute::cast` here
        // because the values buffer is fake — but we can prove the
        // *predicate* allows downcast by hitting the conversion path
        // and seeing the cast attempt error (which is the boundary
        // signal). For a real-data boundary check, the integration
        // test exercises the full pipeline.
        assert!(super::fits_in_narrow_offsets(batch.column(0)));
    }

    #[test]
    fn fits_predicate_is_true_for_non_wide_columns() {
        let f32_arr: ArrayRef = Arc::new(Float32Array::from(vec![1.0_f32, 2.0]));
        assert!(super::fits_in_narrow_offsets(&f32_arr));

        let utf8_arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        assert!(super::fits_in_narrow_offsets(&utf8_arr));
    }

    #[test]
    fn mixed_columns_opportunistic_downcast_partial() {
        // One overflowing LargeUtf8 column + one normal-sized
        // LargeUtf8 column. The first stays wide, the second narrows.
        let overflow = i32::MAX as i64 + 1;
        let big = fake_wide_array(DataType::LargeUtf8, &[0, overflow]);
        let small: ArrayRef = Arc::new(LargeStringArray::from(vec!["x"]));
        let batch = batch_from(vec![
            ("big", DataType::LargeUtf8, big),
            ("small", DataType::LargeUtf8, small),
        ]);

        let downcast = downcast_large_types(&batch).unwrap();
        assert_eq!(downcast.schema().field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(downcast.schema().field(1).data_type(), &DataType::Utf8);
    }
}
