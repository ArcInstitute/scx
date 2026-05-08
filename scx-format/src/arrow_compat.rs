//! Arrow IPC serialization compatibility shims.
//!
//! Arrow IPC `Utf8` and `Binary` columns use 32-bit signed offsets, capping
//! any single string/binary buffer at `2^31 − 1` ≈ 2.15 GB. SCX files with
//! many millions of rows and string-heavy obs metadata overflow this limit
//! during `arrow::ipc::writer::FileWriter::write()` with errors like
//! `Offset overflow error: 2162454931`.
//!
//! [`upcast_to_large_types`] widens `Utf8 → LargeUtf8` and `Binary →
//! LargeBinary` (which use 64-bit offsets) before IPC serialization.
//! [`downcast_large_types`] is the inverse — applied after deserialization
//! so callers always see the canonical narrow types in memory regardless
//! of the on-disk encoding.

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::error::Result;

/// Upcast `Utf8 → LargeUtf8` and `Binary → LargeBinary` (including
/// `Dictionary(_, Utf8|Binary)` value types) so Arrow IPC uses 64-bit
/// offsets. No-op when no narrow string/binary columns are present.
pub fn upcast_to_large_types(batch: &RecordBatch) -> Result<RecordBatch> {
    convert(batch, true)
}

/// Downcast `LargeUtf8 → Utf8` and `LargeBinary → Binary` (including
/// `Dictionary(_, LargeUtf8|LargeBinary)` value types) after IPC
/// deserialization. No-op when no wide string/binary columns are present.
pub fn downcast_large_types(batch: &RecordBatch) -> Result<RecordBatch> {
    convert(batch, false)
}

fn convert(batch: &RecordBatch, widen: bool) -> Result<RecordBatch> {
    let schema = batch.schema();
    let target_for = |dt: &DataType| -> Option<DataType> {
        let inner = match dt {
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
        Some(match dt {
            DataType::Dictionary(k, _) => DataType::Dictionary(k.clone(), Box::new(new_inner)),
            _ => new_inner,
        })
    };

    if !schema
        .fields()
        .iter()
        .any(|f| target_for(f.data_type()).is_some())
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
        if let Some(target_dt) = target_for(field.data_type()) {
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
}
