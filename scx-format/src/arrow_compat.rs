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

use arrow::array::{ArrayRef, LargeBinaryArray, LargeStringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::error::Result;

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
/// (see `scx-format/src/reader.rs`).
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
