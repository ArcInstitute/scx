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
//! ## Why this writes the group itself rather than reusing the obs/var writer
//!
//! The first version of this module built an Arrow `RecordBatch` and handed it
//! to [`crate::h5ad::column_stream::write_dataframe_group_at`], on the argument
//! that the crate should own the h5ad dataframe layout exactly once. Review
//! found three separate defects, all one root cause — that writer models an
//! **obs/var** column, and a `uns` column is a different thing:
//!
//! - It writes *every* `bool` as `encoding-type: "nullable-boolean"` and any
//!   `object` column carrying a null as `nullable-string-array`, because an
//!   obs column is nullable by nature. This module's own ingest arm has no
//!   lossless reading for those, so a frame SCX had just written could not be
//!   read back: `to_h5ad` → `from_h5ad` silently lost every `bool` and
//!   null-bearing string column.
//! - It classifies a `Dictionary(Int32, Boolean)` field `Unsupported`, drops
//!   it, and keeps going — so a boolean categorical exported as an *empty*
//!   DataFrame while this arm still reported success.
//! - It resolves the index by field *name*, so a frame whose index name equals
//!   one of its column names — legal pandas — collided on `create_dataset` and
//!   aborted the entire `to_h5ad` with an opaque HDF5 error.
//!
//! A `uns` frame is one in-memory rectangular table of 1-D columns. Writing
//! that layout directly is *less* code than the batch construction it replaces
//! (no dtype-widening table, no `DictionaryArray`, no field metadata), and it
//! keeps every column's **exact** dtype: an `int8` column lands as `int8`,
//! where the obs/var encodings would have widened it to `int32`.
//!
//! ## Contract: write the whole frame, or decline the whole frame
//!
//! Every column is decoded and checked *before* the group is created. If any
//! one of them has no faithful anndata spelling, this arm declines
//! (`Ok(false)`) and emits [`ConvertWarning::UnsExportedAsRawEnvelope`] — the
//! caller then writes the raw envelope subgroup, which **preserves all the
//! data** (as a dict) rather than dropping a column from an
//! otherwise-successful frame. Nothing is written before that decision, so a
//! decline can never leave a half-built group behind, and no Arrow or HDF5
//! error escapes past this arm to abort an unrelated export.

use hdf5::types::VarLenUnicode;

use crate::h5_write_util::vlu;
use crate::pipeline::ConvertError;
use crate::warnings::{ConvertWarning, WarningSink};

use super::read::SCX_UNS_TYPE_KEY;

/// One decoded column, in the shapes anndata actually spells inside a `uns`
/// dataframe group.
enum UnsColumn {
    /// A numeric or boolean 1-D array: numpy `dtype.str` plus its raw
    /// little-endian bytes, handed to the shared typed-dataset emitter so the
    /// exact width survives.
    Numeric { dtype: String, bytes: Vec<u8> },
    /// A variable-length UTF-8 string array. Nulls are not representable here —
    /// a plain h5ad string dataset has no null — so a column carrying one
    /// declines the frame rather than inventing an empty string for it.
    Strings(Vec<String>),
    /// `categories` + `codes` + the `ordered` attribute. A negative code is
    /// pandas' null and is written through unchanged, exactly as anndata does.
    Categorical {
        categories: Box<UnsColumn>,
        codes: Vec<i32>,
        ordered: bool,
    },
}

impl UnsColumn {
    /// Element count, used to reject a frame whose columns disagree with its
    /// index before anything is written.
    fn len(&self) -> Option<usize> {
        match self {
            UnsColumn::Numeric { dtype, bytes } => {
                let width = dtype_width(dtype)?;
                (bytes.len() % width == 0).then(|| bytes.len() / width)
            }
            UnsColumn::Strings(v) => Some(v.len()),
            UnsColumn::Categorical { codes, .. } => Some(codes.len()),
        }
    }

    fn write(&self, group: &hdf5::Group, name: &str) -> Result<(), ConvertError> {
        match self {
            UnsColumn::Numeric { dtype, bytes } => {
                let n = self.len().unwrap_or(0);
                // Shares the dtype dispatch with the plain-`ndarray` envelope
                // arm, so a column and a standalone array of the same dtype
                // land as the same HDF5 type. `Ok(false)` is unreachable here:
                // `decode_column` accepted this dtype through the same helper.
                if !super::uns::write_uns_envelope_dataset(group, name, dtype, &[n], bytes)? {
                    return Err(ConvertError::Other(format!(
                        "uns DataFrame column '{name}': dtype {dtype} passed preflight but could not be written"
                    )));
                }
                stamp_encoding(&group.dataset(name)?, "array")
            }
            UnsColumn::Strings(values) => {
                let data: Vec<VarLenUnicode> = values.iter().map(|s| vlu(s)).collect();
                let ds = group
                    .new_dataset::<VarLenUnicode>()
                    .shape([data.len()])
                    .create(name)?;
                ds.write(&data)?;
                stamp_encoding(&ds, "string-array")
            }
            UnsColumn::Categorical {
                categories,
                codes,
                ordered,
            } => {
                let cat = group.create_group(name)?;
                categories.write(&cat, "categories")?;
                let codes_ds = cat
                    .new_dataset::<i32>()
                    .shape([codes.len()])
                    .create("codes")?;
                codes_ds.write(codes)?;
                stamp_encoding(&codes_ds, "array")?;
                cat.new_attr::<VarLenUnicode>()
                    .create("encoding-type")?
                    .write_scalar(&vlu("categorical"))?;
                cat.new_attr::<VarLenUnicode>()
                    .create("encoding-version")?
                    .write_scalar(&vlu("0.2.0"))?;
                cat.new_attr::<bool>()
                    .create("ordered")?
                    .write_scalar(ordered)?;
                Ok(())
            }
        }
    }
}

/// Stamp anndata's element encoding on a child dataset.
///
/// anndata's own writer marks every dataframe child (`array` for numeric, bool
/// and categorical codes; `string-array` for strings and string categories),
/// and reads an unmarked one only under an `OldFormatWarning`. The first
/// direct-writer revision omitted these and produced eight such warnings per
/// file — the obs/var writer this module replaced had been stamping them.
fn stamp_encoding(ds: &hdf5::Dataset, encoding: &str) -> Result<(), ConvertError> {
    ds.new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu(encoding))?;
    ds.new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.2.0"))?;
    Ok(())
}

/// Byte width from a numpy `dtype.str` label (`"<f8"` -> 8, `"|b1"` -> 1).
///
/// The label is byte-order char, kind char, then itemsize — the same three-part
/// shape `write_uns_envelope_dataset` parses, kept in step with it.
fn dtype_width(dtype: &str) -> Option<usize> {
    let mut chars = dtype.chars();
    let _order = chars.next()?;
    let _kind = chars.next()?;
    chars.as_str().parse::<usize>().ok()
}

/// Decode a `pandas.DataFrame` envelope and write it as an anndata dataframe
/// group under `parent/name`.
///
/// `Ok(false)` — never an error — for anything this cannot faithfully write,
/// so the caller falls back to the raw-subgroup write with all the data
/// intact. See the module docs for why that is the whole-frame decision rather
/// than a per-column one.
pub(super) fn try_write_uns_dataframe(
    parent: &hdf5::Group,
    name: &str,
    key_path: &str,
    map: &serde_json::Map<String, serde_json::Value>,
    sink: &mut WarningSink,
) -> Result<bool, ConvertError> {
    if map.get(SCX_UNS_TYPE_KEY).and_then(|v| v.as_str()) != Some("pandas.DataFrame") {
        return Ok(false);
    }
    let frame = match decode_frame(map) {
        Ok(f) => f,
        Err(reason) => {
            sink.emit(ConvertWarning::UnsExportedAsRawEnvelope {
                // The accumulated path, not the local group name: a demoted
                // `uns['rank_genes_groups']['pts']` reported as `uns['pts']`
                // names a key the user cannot find.
                key: key_path.to_string(),
                reason,
            });
            return Ok(false);
        }
    };
    frame.write(parent, name)?;
    Ok(true)
}

/// A frame that passed preflight: nothing here can fail to write.
struct DecodedFrame {
    /// On-disk name of the index dataset, and the value of the group's
    /// `_index` attribute.
    index_name: String,
    index: UnsColumn,
    columns: Vec<(String, UnsColumn)>,
}

impl DecodedFrame {
    fn write(&self, parent: &hdf5::Group, name: &str) -> Result<(), ConvertError> {
        let group = parent.create_group(name)?;
        group
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")?
            .write_scalar(&vlu("dataframe"))?;
        group
            .new_attr::<VarLenUnicode>()
            .create("encoding-version")?
            .write_scalar(&vlu("0.2.0"))?;
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&vlu(&self.index_name))?;

        // `column-order` carries the frame's column order; anndata falls back
        // to HDF5 member order (alphabetical) without it. Written with an
        // explicit shape so the column-less case is a length-0 array rather
        // than a scalar, which anndata rejects.
        let order: Vec<VarLenUnicode> = self.columns.iter().map(|(n, _)| vlu(n)).collect();
        group
            .new_attr::<VarLenUnicode>()
            .shape([order.len()])
            .create("column-order")?
            .write_raw(&order)?;

        self.index.write(&group, &self.index_name)?;
        for (col_name, col) in &self.columns {
            col.write(&group, col_name)?;
        }
        Ok(())
    }
}

/// Envelope → a frame every part of which is known writable, or `Err(reason)`
/// naming what could not be spelled. The reason travels into the warning, so
/// it has to read as a sentence fragment.
fn decode_frame(map: &serde_json::Map<String, serde_json::Value>) -> Result<DecodedFrame, String> {
    let Some(serde_json::Value::Array(col_names)) = map.get("columns") else {
        return Err("envelope has no `columns` list".into());
    };
    let Some(serde_json::Value::Object(data)) = map.get("data") else {
        return Err("envelope has no `data` object".into());
    };
    let Some(serde_json::Value::Object(index_env)) = map.get("index") else {
        return Err("envelope has no `index` object".into());
    };

    // The index rides in a `pandas.Index` envelope, whose `data` is the array
    // envelope proper.
    let index = index_env
        .get("data")
        .ok_or_else(|| "index envelope has no `data`".to_string())
        .and_then(|v| decode_column(v).map_err(|e| format!("index: {e}")))?;
    // `null` (and a missing key) is genuinely unnamed and becomes anndata's
    // `_index`. A *non-string* scalar name — `df.index.name = 7`, legal pandas —
    // used to fall into that same arm and be silently erased; anndata itself
    // refuses such a name, so declining says so instead of quietly dropping it.
    let index_name = match index_env.get("name") {
        Some(serde_json::Value::String(s)) => s.clone(),
        None | Some(serde_json::Value::Null) => "_index".to_string(),
        Some(other) => {
            return Err(format!(
                "index name {other} is not a string, and h5ad stores the index name as an \
                 HDF5 member name"
            ))
        }
    };
    if !super::uns::is_safe_hdf5_member_name(&index_name) {
        return Err(format!(
            "index name {index_name:?} is not usable as an HDF5 member name"
        ));
    }

    let n_rows = index
        .len()
        .ok_or_else(|| "index length is not determinable".to_string())?;

    let mut columns: Vec<(String, UnsColumn)> = Vec::with_capacity(col_names.len());
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for col in col_names {
        let serde_json::Value::String(col) = col else {
            return Err("a `columns` entry is not a string".into());
        };
        if !super::uns::is_safe_hdf5_member_name(col) {
            return Err(format!(
                "column name {col:?} is not usable as an HDF5 member name"
            ));
        }
        // pyscx's encoder refuses duplicates, so a frame it wrote cannot get
        // here — but an envelope from anywhere else can, and `data` is a JSON
        // object, so the second copy would `create_dataset` a name that already
        // exists and abort the export from inside the write.
        if !seen.insert(col.as_str()) {
            return Err(format!("duplicate column name {col:?} in `columns`"));
        }
        // A column named like the index would collide on `create_dataset` and
        // abort the export. Legal in pandas (`df.index.name == "gene"` with a
        // "gene" column), so decline rather than fail.
        if *col == index_name {
            return Err(format!(
                "column '{col}' has the same name as the index, which anndata stores in the same group"
            ));
        }
        let value = data.get(col.as_str()).ok_or_else(|| {
            format!("column '{col}' is listed in `columns` but missing from `data`")
        })?;
        let decoded = decode_column(value).map_err(|e| format!("column '{col}': {e}"))?;
        if decoded.len() != Some(n_rows) {
            return Err(format!(
                "column '{col}' has {:?} values but the index has {n_rows}",
                decoded.len()
            ));
        }
        columns.push((col.clone(), decoded));
    }

    Ok(DecodedFrame {
        index_name,
        index,
        columns,
    })
}

/// One envelope → a writable column, or `Err(reason)`.
fn decode_column(value: &serde_json::Value) -> Result<UnsColumn, String> {
    let serde_json::Value::Object(map) = value else {
        return Err("not an envelope object".into());
    };
    match map.get(SCX_UNS_TYPE_KEY).and_then(|v| v.as_str()) {
        Some("ndarray") => decode_ndarray(map),
        Some("categorical") => decode_categorical(map),
        Some(other) => Err(format!("unsupported `{SCX_UNS_TYPE_KEY}` '{other}'")),
        None => Err("not a tagged envelope".into()),
    }
}

/// A `categorical` envelope → `categories` + `codes` + `ordered`.
///
/// Codes widen to `i32`, which is the width anndata's `codes` dataset uses and
/// the one this crate's reader expects; a negative code is pandas' null and is
/// preserved.
///
/// Boolean categories are **not** refused. An earlier revision did, on the
/// premise that no reader here takes them back — measured false: this crate's
/// `uns_dataframe_column_envelope` has a `TypeDescriptor::Boolean` arm, and an
/// anndata-written ordered boolean categorical ingests with its values,
/// categories and `ordered` bit intact. The guard was a false refusal that
/// demoted a valid frame.
fn decode_categorical(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<UnsColumn, String> {
    let (Some(categories), Some(codes)) = (map.get("categories"), map.get("codes")) else {
        return Err("categorical envelope is missing `categories` or `codes`".into());
    };
    let categories = decode_column(categories).map_err(|e| format!("categories: {e}"))?;

    let codes = match decode_column(codes)? {
        UnsColumn::Numeric { dtype, bytes } => codes_to_i32(&dtype, &bytes)
            .ok_or_else(|| format!("codes dtype {dtype} is not an integer width"))?,
        _ => return Err("categorical `codes` is not a numeric array".into()),
    };
    let ordered = map
        .get("ordered")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(UnsColumn::Categorical {
        categories: Box::new(categories),
        codes,
        ordered,
    })
}

/// Categorical codes, whatever signed width pandas used, as `i32`.
fn codes_to_i32(dtype: &str, bytes: &[u8]) -> Option<Vec<i32>> {
    macro_rules! widen {
        ($ty:ty, $w:expr) => {{
            if bytes.len() % $w != 0 {
                return None;
            }
            Some(
                bytes
                    .chunks_exact($w)
                    .map(|c| <$ty>::from_le_bytes(c.try_into().unwrap()) as i32)
                    .collect(),
            )
        }};
    }
    match (dtype.chars().nth(1), dtype_width(dtype)?) {
        (Some('i'), 1) => Some(bytes.iter().map(|&b| b as i8 as i32).collect()),
        (Some('i'), 2) => widen!(i16, 2),
        (Some('i'), 4) => widen!(i32, 4),
        (Some('i'), 8) => widen!(i64, 8),
        _ => None,
    }
}

/// An `ndarray` envelope → a writable column.
///
/// Only 1-D arrays are dataframe columns, so a higher-rank envelope declines
/// rather than flattening silently. Numeric payloads pass their bytes through
/// untouched, dtype and all — no widening.
fn decode_ndarray(map: &serde_json::Map<String, serde_json::Value>) -> Result<UnsColumn, String> {
    let Some(dtype) = map.get("dtype").and_then(|v| v.as_str()) else {
        return Err("ndarray envelope has no `dtype`".into());
    };
    let Some(serde_json::Value::Array(shape)) = map.get("shape") else {
        return Err("ndarray envelope has no `shape`".into());
    };
    if shape.len() != 1 {
        return Err(format!("{}-dimensional, not a column", shape.len()));
    }
    let Some(n) = shape[0].as_u64().map(|n| n as usize) else {
        return Err("`shape` entry is not a non-negative integer".into());
    };

    match map.get("encoding").and_then(|v| v.as_str()) {
        Some("json") => {
            let Some(serde_json::Value::Array(items)) = map.get("data") else {
                return Err("json-encoded ndarray has no `data` list".into());
            };
            let mut out: Vec<String> = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    serde_json::Value::String(s) => out.push(s.clone()),
                    // A plain h5ad string dataset has no null. Rather than
                    // write `""` and read back the wrong value, decline the
                    // frame so the raw envelope keeps the null.
                    serde_json::Value::Null => {
                        return Err(
                            "contains a null, which a plain h5ad string dataset cannot hold".into(),
                        )
                    }
                    _ => return Err("a string-array element is not a string".into()),
                }
            }
            if out.len() != n {
                return Err(format!("declares {n} elements but carries {}", out.len()));
            }
            Ok(UnsColumn::Strings(out))
        }
        Some("base64le") => {
            use base64::Engine;
            let Some(b64) = map.get("data").and_then(|v| v.as_str()) else {
                return Err("base64 ndarray has no `data` string".into());
            };
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|_| "`data` is not valid base64".to_string())?;
            // Byte order is explicit in the label; reinterpreting a big-endian
            // payload as little-endian would silently corrupt every value.
            let order = dtype.chars().next();
            if !matches!(order, Some('<') | Some('|') | Some('=')) {
                return Err(format!("dtype {dtype} is not little-endian"));
            }
            let width = dtype_width(dtype).ok_or_else(|| format!("unparseable dtype {dtype}"))?;
            if width == 0 || bytes.len() != n * width {
                return Err(format!(
                    "dtype {dtype} and {n} elements do not match {} payload bytes",
                    bytes.len()
                ));
            }
            // Reject here rather than at write time: `write_uns_envelope_dataset`
            // answers `Ok(false)` for a kind it cannot emit, and by then the
            // group would already exist.
            if !matches!(
                (dtype.chars().nth(1), width),
                (Some('f'), 2 | 4 | 8)
                    | (Some('i'), 1 | 2 | 4 | 8)
                    | (Some('u'), 1 | 2 | 4 | 8)
                    | (Some('b'), 1)
            ) {
                return Err(format!("dtype {dtype} has no HDF5 form"));
            }
            Ok(UnsColumn::Numeric {
                dtype: dtype.to_string(),
                bytes,
            })
        }
        other => Err(format!("unsupported ndarray encoding {other:?}")),
    }
}

#[cfg(test)]
#[path = "uns_dataframe_tests.rs"]
mod tests;
