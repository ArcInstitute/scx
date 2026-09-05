//! `/uns` export, the `pandas.DataFrame` arm: envelope JSON → an anndata
//! dataframe group.
//!
//! Every other `__scx_type__` envelope the exporter cannot decode natively
//! falls back to a raw subgroup carrying its JSON keys as datasets, which
//! anndata reads back as a plain dict. For a frame that is not good enough:
//! `sc.tl.filter_rank_genes_groups` does
//! `uns[key]["pts"][group].loc[var_names]` and breaks on a dict, which is the
//! whole reason the envelope exists.
//!
//! The group layout — `encoding-type`, `_index`, `column-order`, per-column
//! datasets, categoricals as a `categories`/`codes` sub-group with an
//! `ordered` attribute — is **not** re-implemented here. This module turns the
//! envelope into an Arrow [`RecordBatch`] and hands it to
//! [`crate::h5ad::column_stream::write_dataframe_group_at`], the crate's one
//! dataframe-column writer, so the nine column encodings keep existing exactly
//! once (see `h5ad/mod.rs`).
//!
//! **Dtype widening.** Those nine encodings are i32 / i64 / f32 / f64 / Utf8 /
//! Boolean / Categorical / nullable / unsupported, so an `int8`, `uint16` or
//! `float16` column lands in h5ad as `int32` / `int32` / `float32`, and a
//! `bool` or `object` column lands in one of anndata's nullable encodings
//! (read back as pandas `boolean` / `string`-like rather than numpy `bool` /
//! `object`). That is what obs and var already do; the SCX file and every
//! pyscx round trip keep the exact dtype, and only this export widens.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};

use crate::pipeline::ConvertError;
use crate::warnings::WarningSink;
use crate::CATEGORICAL_ORDERED_KEY;

/// The on-disk name anndata gives an unnamed dataframe index.
///
/// `write_dataframe_header` resolves the index from pandas `index_columns`
/// schema metadata, else from a field literally called `__index_level_0__` or
/// `_index`, else field 0 — so naming an unnamed index this way is what makes
/// it land as `_index` rather than as a column.
const UNNAMED_INDEX_FIELD: &str = "__index_level_0__";

/// Decode a `pandas.DataFrame` envelope and write it as an anndata dataframe
/// group under `parent/name`.
///
/// `Ok(false)` — never an error — for anything this cannot faithfully decode,
/// so the caller falls back to the raw-subgroup write. Refusing to export the
/// whole file over one odd column would be a worse trade than the pre-X6
/// behaviour it replaces.
pub(super) fn try_write_uns_dataframe(
    parent: &hdf5::Group,
    name: &str,
    map: &serde_json::Map<String, serde_json::Value>,
    sink: &mut WarningSink,
) -> Result<bool, ConvertError> {
    if map
        .get(super::read::SCX_UNS_TYPE_KEY)
        .and_then(|v| v.as_str())
        != Some("pandas.DataFrame")
    {
        return Ok(false);
    }
    let Some(batch) = envelope_to_record_batch(map)? else {
        return Ok(false);
    };
    super::column_stream::write_dataframe_group_at(parent, name, &batch, sink)?;
    Ok(true)
}

/// Envelope → `RecordBatch` with the index as field 0.
///
/// `None` when a leaf cannot be decoded. Column order comes from the
/// envelope's `columns` list, never from `data`'s member order, because JSON
/// object order is not a contract — and `column-order` on the written group
/// follows Arrow field order, so getting this wrong would reorder the frame
/// anndata reads back.
fn envelope_to_record_batch(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<RecordBatch>, ConvertError> {
    let Some(serde_json::Value::Array(col_names)) = map.get("columns") else {
        return Ok(None);
    };
    let Some(serde_json::Value::Object(data)) = map.get("data") else {
        return Ok(None);
    };
    let Some(serde_json::Value::Object(index_env)) = map.get("index") else {
        return Ok(None);
    };

    // The index rides in a `pandas.Index` envelope, whose `data` is the array
    // envelope proper.
    let Some(index_data) = index_env.get("data") else {
        return Ok(None);
    };
    let Some((index_array, _)) = decode_column(index_data)? else {
        return Ok(None);
    };
    let index_name = match index_env.get("name") {
        Some(serde_json::Value::String(s)) => s.clone(),
        _ => UNNAMED_INDEX_FIELD.to_string(),
    };

    let n_rows = index_array.len();
    let mut fields: Vec<Field> = vec![Field::new(
        index_name,
        index_array.data_type().clone(),
        false,
    )];
    let mut arrays: Vec<ArrayRef> = vec![index_array];

    for col in col_names {
        let serde_json::Value::String(col) = col else {
            return Ok(None);
        };
        let Some(value) = data.get(col.as_str()) else {
            return Ok(None);
        };
        let Some((array, ordered)) = decode_column(value)? else {
            return Ok(None);
        };
        // A column shorter or longer than the index would build an invalid
        // batch; a malformed envelope must degrade to the raw-subgroup write,
        // not abort the export.
        if array.len() != n_rows {
            return Ok(None);
        }
        let mut field = Field::new(col.clone(), array.data_type().clone(), true);
        if let Some(ordered) = ordered {
            // The only supplier of the `ordered` bit `finalize_column_writer`
            // writes back out as the group's attribute.
            field = field.with_metadata(HashMap::from([(
                CATEGORICAL_ORDERED_KEY.to_string(),
                ordered.to_string(),
            )]));
        }
        fields.push(field);
        arrays.push(array);
    }

    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| ConvertError::Other(format!("uns DataFrame export: {e}")))?;
    Ok(Some(batch))
}

/// One envelope → an Arrow array, plus the categorical `ordered` bit when the
/// column is one. `None` for a leaf with no faithful Arrow spelling.
fn decode_column(
    value: &serde_json::Value,
) -> Result<Option<(ArrayRef, Option<bool>)>, ConvertError> {
    let serde_json::Value::Object(map) = value else {
        return Ok(None);
    };
    match map
        .get(super::read::SCX_UNS_TYPE_KEY)
        .and_then(|v| v.as_str())
    {
        Some("ndarray") => Ok(decode_ndarray(map)?.map(|a| (a, None))),
        Some("categorical") => decode_categorical(map),
        _ => Ok(None),
    }
}

/// A `categorical` envelope → an Arrow `DictionaryArray<Int32Type>`.
///
/// Codes arrive as whatever integer width pandas used (`int8` for a few
/// levels); Arrow dictionaries are `Int32`, and `create_column_writer` builds
/// its `codes` dataset as `i32`, so the widening here is the on-disk form
/// either way. A negative code is pandas' null and becomes an Arrow null.
fn decode_categorical(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<(ArrayRef, Option<bool>)>, ConvertError> {
    let (Some(categories), Some(codes)) = (map.get("categories"), map.get("codes")) else {
        return Ok(None);
    };
    let Some(values) = decode_ndarray_map(categories)? else {
        return Ok(None);
    };
    let Some(codes) = decode_ndarray_map(codes)? else {
        return Ok(None);
    };
    let ordered = map
        .get("ordered")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let keys: Int32Array = match codes.data_type() {
        DataType::Int32 => codes
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("checked")
            .iter()
            .map(|c| c.filter(|&c| c >= 0))
            .collect(),
        DataType::Int64 => codes
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("checked")
            .iter()
            .map(|c| c.filter(|&c| c >= 0).map(|c| c as i32))
            .collect(),
        _ => return Ok(None),
    };

    let dict = DictionaryArray::<Int32Type>::try_new(keys, values)
        .map_err(|e| ConvertError::Other(format!("uns DataFrame categorical export: {e}")))?;
    Ok(Some((Arc::new(dict), Some(ordered))))
}

fn decode_ndarray_map(value: &serde_json::Value) -> Result<Option<ArrayRef>, ConvertError> {
    let serde_json::Value::Object(map) = value else {
        return Ok(None);
    };
    if map
        .get(super::read::SCX_UNS_TYPE_KEY)
        .and_then(|v| v.as_str())
        != Some("ndarray")
    {
        return Ok(None);
    }
    decode_ndarray(map)
}

/// An `ndarray` envelope → an Arrow array.
///
/// `encoding: "json"` carries object/string arrays as a JSON list; everything
/// else is base64 little-endian raw bytes, reinterpreted per the numpy
/// `dtype.str` label. Only 1-D arrays make sense as dataframe columns, so a
/// higher-rank envelope declines rather than flattening silently.
fn decode_ndarray(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<ArrayRef>, ConvertError> {
    let Some(dtype) = map.get("dtype").and_then(|v| v.as_str()) else {
        return Ok(None);
    };
    let Some(serde_json::Value::Array(shape)) = map.get("shape") else {
        return Ok(None);
    };
    if shape.len() != 1 {
        return Ok(None);
    }
    let Some(n) = shape[0].as_u64().map(|n| n as usize) else {
        return Ok(None);
    };

    match map.get("encoding").and_then(|v| v.as_str()) {
        Some("json") => {
            let Some(serde_json::Value::Array(items)) = map.get("data") else {
                return Ok(None);
            };
            let mut out: Vec<Option<String>> = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    serde_json::Value::String(s) => out.push(Some(s.clone())),
                    serde_json::Value::Null => out.push(None),
                    // A nested list means a >1-D object array; not a column.
                    _ => return Ok(None),
                }
            }
            Ok(Some(Arc::new(StringArray::from(out)) as ArrayRef))
        }
        Some("base64le") => {
            use base64::Engine;
            let Some(b64) = map.get("data").and_then(|v| v.as_str()) else {
                return Ok(None);
            };
            let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else {
                return Ok(None);
            };
            Ok(numeric_array(dtype, &bytes, n))
        }
        _ => Ok(None),
    }
}

/// Little-endian `bytes` reinterpreted as `dtype`, widened into whichever of
/// the four numeric column encodings can hold it losslessly.
///
/// `uint64` has no lossless target — `i64` cannot hold its upper half — so it
/// declines rather than wrapping a large count into a negative one.
fn numeric_array(dtype: &str, bytes: &[u8], n: usize) -> Option<ArrayRef> {
    /// Reinterpret `bytes` as `n` little-endian values of a fixed width.
    macro_rules! read_le {
        ($ty:ty, $w:expr) => {{
            if bytes.len() != n * $w {
                return None;
            }
            bytes
                .chunks_exact($w)
                .map(|c| <$ty>::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<$ty>>()
        }};
    }

    // Byte order is explicit in the label; a big-endian payload is not
    // something this workspace writes, and reinterpreting it as LE would
    // silently corrupt every value.
    let arr: ArrayRef = match dtype {
        "|b1" | "<b1" | "=b1" => {
            if bytes.len() != n {
                return None;
            }
            Arc::new(BooleanArray::from(
                bytes.iter().map(|&b| b != 0).collect::<Vec<bool>>(),
            ))
        }
        "|i1" | "<i1" | "=i1" => Arc::new(Int32Array::from(
            read_le!(i8, 1)
                .into_iter()
                .map(i32::from)
                .collect::<Vec<i32>>(),
        )),
        "<i2" | "=i2" => Arc::new(Int32Array::from(
            read_le!(i16, 2)
                .into_iter()
                .map(i32::from)
                .collect::<Vec<i32>>(),
        )),
        "<i4" | "=i4" => Arc::new(Int32Array::from(read_le!(i32, 4))),
        "<i8" | "=i8" => Arc::new(Int64Array::from(read_le!(i64, 8))),
        "|u1" | "<u1" | "=u1" => Arc::new(Int32Array::from(
            read_le!(u8, 1)
                .into_iter()
                .map(i32::from)
                .collect::<Vec<i32>>(),
        )),
        "<u2" | "=u2" => Arc::new(Int32Array::from(
            read_le!(u16, 2)
                .into_iter()
                .map(i32::from)
                .collect::<Vec<i32>>(),
        )),
        "<u4" | "=u4" => Arc::new(Int64Array::from(
            read_le!(u32, 4)
                .into_iter()
                .map(i64::from)
                .collect::<Vec<i64>>(),
        )),
        "<f4" | "=f4" => Arc::new(Float32Array::from(read_le!(f32, 4))),
        "<f8" | "=f8" => Arc::new(Float64Array::from(read_le!(f64, 8))),
        "<f2" | "=f2" => {
            if bytes.len() != n * 2 {
                return None;
            }
            Arc::new(Float32Array::from(
                bytes
                    .chunks_exact(2)
                    .map(|c| f32::from(half::f16::from_le_bytes([c[0], c[1]])))
                    .collect::<Vec<f32>>(),
            ))
        }
        _ => return None,
    };
    Some(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The envelope a pyscx-written frame actually looks like, so the test
    /// exercises the shape the decoder meets in the wild rather than one
    /// invented to suit it.
    fn frame_envelope() -> serde_json::Value {
        serde_json::json!({
            "__scx_type__": "pandas.DataFrame",
            "index": {
                "__scx_type__": "pandas.Index",
                "name": "row",
                "data": {
                    "__scx_type__": "ndarray", "dtype": "object", "shape": [2],
                    "encoding": "json", "data": ["r0", "r1"],
                },
            },
            // Deliberately not alphabetical: `column-order` on the written
            // group follows Arrow field order, so a decoder that iterated
            // `data`'s members instead would reorder the frame silently.
            "columns": ["zed", "abe", "grade"],
            "data": {
                // [1.0, 2.0] as little-endian f64.
                "zed": {
                    "__scx_type__": "ndarray", "dtype": "<f8", "shape": [2],
                    "encoding": "base64le", "data": "AAAAAAAA8D8AAAAAAAAAQA==",
                },
                // [3, 4] as little-endian i64.
                "abe": {
                    "__scx_type__": "ndarray", "dtype": "<i8", "shape": [2],
                    "encoding": "base64le", "data": "AwAAAAAAAAAEAAAAAAAAAA==",
                },
                "grade": {
                    "__scx_type__": "categorical",
                    "ordered": true,
                    // codes [1, 0] as int8.
                    "codes": {
                        "__scx_type__": "ndarray", "dtype": "|i1", "shape": [2],
                        "encoding": "base64le", "data": "AQA=",
                    },
                    "categories": {
                        "__scx_type__": "ndarray", "dtype": "object", "shape": [3],
                        "encoding": "json", "data": ["lo", "mid", "hi"],
                    },
                },
            },
        })
    }

    /// The frame arm writes the anndata dataframe layout, not a raw subgroup.
    ///
    /// Asserting the on-disk attributes rather than a Python round trip is the
    /// point: `encoding-type`, `_index` and `column-order` are exactly what
    /// anndata dispatches on, and they are what the pre-X6 generic path never
    /// wrote.
    #[test]
    fn frame_envelope_writes_an_anndata_dataframe_group() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("t.h5")).unwrap();
        let uns = file.create_group("uns").unwrap();
        let mut sink = crate::warnings::WarningSink::log();

        let env = frame_envelope();
        assert!(
            try_write_uns_dataframe(&uns, "tbl", env.as_object().unwrap(), &mut sink).unwrap(),
            "a well-formed frame envelope must be claimed by this arm"
        );

        let g = uns.group("tbl").unwrap();
        let read_attr = |name: &str| -> String {
            g.attr(name)
                .unwrap()
                .read_scalar::<hdf5::types::VarLenUnicode>()
                .unwrap()
                .to_string()
        };
        assert_eq!(read_attr("encoding-type"), "dataframe");
        assert_eq!(read_attr("_index"), "row");

        let order: Vec<String> = g
            .attr("column-order")
            .unwrap()
            .read_1d::<hdf5::types::VarLenUnicode>()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            order,
            ["zed", "abe", "grade"],
            "column-order must follow the envelope's `columns`, not `data`'s member order"
        );

        assert_eq!(g.dataset("row").unwrap().shape(), [2]);
        assert_eq!(g.dataset("zed").unwrap().shape(), [2]);

        // The categorical becomes a real categorical group, `ordered` included
        // — the bit the flatten path used to drop.
        let cat = g.group("grade").unwrap();
        assert_eq!(
            cat.attr("encoding-type")
                .unwrap()
                .read_scalar::<hdf5::types::VarLenUnicode>()
                .unwrap()
                .to_string(),
            "categorical"
        );
        assert!(cat.attr("ordered").unwrap().read_scalar::<bool>().unwrap());
        assert_eq!(
            cat.dataset("codes").unwrap().read_raw::<i32>().unwrap(),
            [1, 0]
        );
        // Declared categories survive whole: no row uses "mid", and ingest
        // applies no row filter, so pruning it would be wrong.
        let cats: Vec<String> = cat
            .dataset("categories")
            .unwrap()
            .read_1d::<hdf5::types::VarLenUnicode>()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(cats, ["lo", "mid", "hi"]);
    }

    /// An unnamed index must land as `_index`, anndata's own spelling.
    #[test]
    fn unnamed_index_lands_as_underscore_index() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("t.h5")).unwrap();
        let uns = file.create_group("uns").unwrap();
        let mut sink = crate::warnings::WarningSink::log();

        let mut env = frame_envelope();
        env["index"]["name"] = serde_json::Value::Null;
        assert!(try_write_uns_dataframe(&uns, "tbl", env.as_object().unwrap(), &mut sink).unwrap());

        let g = uns.group("tbl").unwrap();
        assert_eq!(
            g.attr("_index")
                .unwrap()
                .read_scalar::<hdf5::types::VarLenUnicode>()
                .unwrap()
                .to_string(),
            "_index"
        );
        assert!(g.dataset("_index").is_ok());
    }

    /// Every malformed shape declines rather than erroring, so the caller falls
    /// back to the raw-subgroup write.
    ///
    /// Declining, not failing, is the contract: refusing to export a whole file
    /// over one odd `uns` value would be worse than the pre-X6 behaviour this
    /// replaces.
    #[test]
    fn malformed_frame_envelopes_decline_rather_than_erroring() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("t.h5")).unwrap();
        let uns = file.create_group("uns").unwrap();
        let mut sink = crate::warnings::WarningSink::log();

        let cases: Vec<(&str, Box<dyn Fn(&mut serde_json::Value)>)> = vec![
            (
                "wrong_tag",
                Box::new(|e: &mut serde_json::Value| {
                    e["__scx_type__"] = "something.else".into();
                }),
            ),
            (
                "no_columns",
                Box::new(|e: &mut serde_json::Value| {
                    e.as_object_mut().unwrap().remove("columns");
                }),
            ),
            (
                "column_not_in_data",
                Box::new(|e: &mut serde_json::Value| {
                    e["columns"] = serde_json::json!(["zed", "ghost"]);
                }),
            ),
            (
                "length_mismatch",
                Box::new(|e: &mut serde_json::Value| {
                    // One value where the index has two: an invalid RecordBatch.
                    e["data"]["zed"]["shape"] = serde_json::json!([1]);
                    e["data"]["zed"]["data"] = "AAAAAAAA8D8=".into();
                    e["columns"] = serde_json::json!(["zed"]);
                }),
            ),
            (
                "big_endian_column",
                Box::new(|e: &mut serde_json::Value| {
                    e["data"]["zed"]["dtype"] = ">f8".into();
                    e["columns"] = serde_json::json!(["zed"]);
                }),
            ),
            (
                "uint64_column",
                Box::new(|e: &mut serde_json::Value| {
                    // No lossless Arrow/h5ad target; wrapping into i64 would turn a
                    // large count negative.
                    e["data"]["zed"]["dtype"] = "<u8".into();
                    e["columns"] = serde_json::json!(["zed"]);
                }),
            ),
            (
                "two_dimensional_column",
                Box::new(|e: &mut serde_json::Value| {
                    e["data"]["zed"]["shape"] = serde_json::json!([1, 2]);
                    e["columns"] = serde_json::json!(["zed"]);
                }),
            ),
        ];

        for (name, mutate) in cases {
            let mut env = frame_envelope();
            mutate(&mut env);
            assert!(
                !try_write_uns_dataframe(&uns, name, env.as_object().unwrap(), &mut sink).unwrap(),
                "'{name}' must decline, not claim the value"
            );
            assert!(
                uns.group(name).is_err() && uns.dataset(name).is_err(),
                "'{name}' must not have written anything before declining"
            );
        }
    }
}
