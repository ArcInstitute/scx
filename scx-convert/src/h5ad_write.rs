// Write scx file back to h5ad format

use std::collections::HashMap;
use std::path::Path;

use arrow::array::{
    Array, AsArray, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use arrow::datatypes::{
    DataType, Field, Int16Type, Int32Type, Int64Type, Int8Type, Schema, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use hdf5::types::VarLenUnicode;
use ndarray::ArrayView1;

/// Convert &str to VarLenUnicode, validating no NUL bytes are present.
fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap_or_else(|_| {
        // Strip NUL bytes rather than panicking (finding 8.2).
        let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
        cleaned
            .parse::<VarLenUnicode>()
            .expect("cleaned string should have no NUL bytes")
    })
}

use scx_format::reader::ScxReader;

use super::pipeline::ConvertError;
use super::warnings::{ConvertWarning, WarningSink};

/// Write an SCX file to h5ad format.
pub fn write_scx_to_h5ad(
    scx_path: &Path,
    h5ad_path: &Path,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    let file = hdf5::File::create(h5ad_path)?;

    // Honor deletion vectors on every leg (X, obs, layers). The CSR
    // and layer readers filter by DV directly; obs goes through the
    // shared streaming-or-eager dispatcher which applies the same
    // global keep mask. Pre-fix, this path silently dropped DV
    // semantics — both /X and obs were written unfiltered.
    let keep_mask = crate::h5ad_stream_write::build_keep_mask(&reader)?;

    // Read the full CSR matrix (DV-filtered when active)
    let csr = reader.read_all_csr_shards_filtered()?;
    let n_obs = csr.shape.0;
    let n_vars = csr.shape.1;

    // Write X as CSR group
    write_sparse_group(
        &file,
        "X",
        &csr.indptr,
        &csr.indices,
        &csr.data,
        n_obs,
        n_vars,
    )?;

    // Write obs / var via the shared dispatcher so legacy + sharded
    // sources both flow through one code path and the DV keep mask
    // is honored.
    let root = file.as_group()?;
    crate::h5ad_stream_write::write_obs_streaming_or_eager(
        &root,
        &reader,
        keep_mask.as_deref(),
        sink,
    )?;
    crate::h5ad_stream_write::write_var_streaming_or_eager(&root, &reader, sink)?;

    // Write obsm (DV-filtered when active — obs-axis rows must match
    // /X and /obs).
    if let Ok(obsm_map) = reader.read_all_obsm() {
        if !obsm_map.is_empty() {
            let obsm_group = file.create_group("obsm")?;
            for (name, batch) in &obsm_map {
                let filtered = match keep_mask.as_deref() {
                    Some(mask) => {
                        crate::h5ad_stream_write::filter_record_batch_by_mask(batch, mask)?
                    }
                    None => batch.clone(),
                };
                write_obsm_entry(&obsm_group, name, &filtered)?;
            }
        }
    }

    // Write uns
    if let Ok(uns) = reader.read_uns() {
        let uns_group = file.create_group("uns")?;
        write_uns_entries(&uns_group, &uns)?;
    }

    // Write layers (DV-filtered when active — layers share X's
    // row count by AnnData invariant)
    let layer_names = reader.layer_names();
    if !layer_names.is_empty() {
        let layers_group = file.create_group("layers")?;
        for layer_name in &layer_names {
            if let Ok(layer_csr) = reader.read_layer_filtered(layer_name) {
                let lg = layers_group.create_group(layer_name)?;
                write_sparse_arrays(
                    &lg,
                    &layer_csr.indptr,
                    &layer_csr.indices,
                    &layer_csr.data,
                    layer_csr.shape.0,
                    layer_csr.shape.1,
                )?;
            }
        }
    }

    // Write /raw (DV-filtered on the obs axis like /X).
    write_raw_to_h5ad(&root, &reader, keep_mask.as_deref(), sink)?;

    Ok(())
}

/// Write the `adata.raw` group (`raw/X` + `raw/var`) into an output
/// h5ad if the SCX file carries a raw matrix. Raw shares X's obs axis,
/// so the same deletion-vector keep mask is applied to its rows. Shared
/// by the eager and streaming SCX→h5ad export paths.
pub(super) fn write_raw_to_h5ad(
    root: &hdf5::Group,
    reader: &ScxReader,
    keep_mask: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    if !reader.has_raw() {
        return Ok(());
    }
    let raw = reader.read_all_raw_csr_shards()?;
    let raw_n_vars = raw.shape.1;
    let (indptr, indices, data, n_obs) = match keep_mask {
        Some(mask) => filter_csr_rows(&raw.indptr, &raw.indices, &raw.data, mask),
        None => (raw.indptr, raw.indices, raw.data, raw.shape.0),
    };

    let raw_group = root.create_group("raw")?;
    write_sparse_group_at(&raw_group, "X", &indptr, &indices, &data, n_obs, raw_n_vars)?;
    let raw_var = reader.read_raw_var()?;
    write_dataframe_group_at(&raw_group, "var", &raw_var, sink)?;
    Ok(())
}

/// Subset CSR rows by a boolean obs keep-mask, returning new
/// `(indptr, indices, data, n_kept_rows)`. Used to apply deletion
/// vectors to the raw matrix on export (raw shares the obs axis).
fn filter_csr_rows(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    mask: &[bool],
) -> (Vec<i64>, Vec<i32>, Vec<f32>, usize) {
    let n_rows = indptr.len().saturating_sub(1).min(mask.len());
    let mut out_indptr = vec![0i64];
    let mut out_indices = Vec::new();
    let mut out_data = Vec::new();
    for (row, &keep) in mask.iter().enumerate().take(n_rows) {
        if keep {
            let s = indptr[row] as usize;
            let e = indptr[row + 1] as usize;
            out_indices.extend_from_slice(&indices[s..e]);
            out_data.extend_from_slice(&data[s..e]);
            out_indptr.push(out_indices.len() as i64);
        }
    }
    let n = out_indptr.len() - 1;
    (out_indptr, out_indices, out_data, n)
}

fn write_sparse_group(
    file: &hdf5::File,
    name: &str,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    let group = file.create_group(name)?;
    write_sparse_arrays(&group, indptr, indices, data, n_obs, n_vars)
}

// Module-internal helpers for the h5mu writer (Phase D.2). All
// take a parent `hdf5::Group` instead of the root `hdf5::File` so
// per-modality blocks under `/mod/{name}/…` can reuse the same
// emitters as `/X`, `/obs`, `/var`, `/obsm/…`, `/uns/…`.

pub(super) fn write_sparse_group_at(
    parent: &hdf5::Group,
    name: &str,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    let group = parent.create_group(name)?;
    write_sparse_arrays(&group, indptr, indices, data, n_obs, n_vars)
}

pub(super) fn write_dataframe_group_at(
    parent: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let group = parent.group(name).or_else(|_| parent.create_group(name))?;
    write_dataframe_body(&group, name, batch, sink)
}

/// Shared body for `write_dataframe_group_at`. Caller is responsible
/// for opening or creating `group`. Resolves the pandas index from
/// schema metadata, renames pyarrow's `__index_level_0__` to anndata's
/// `_index` literal on disk (named indexes keep their original name),
/// excludes the index column from `column-order`, and always writes
/// `column-order` (length-0 OK) since anndata.read_h5ad requires the
/// attribute to be present.
fn write_dataframe_body(
    group: &hdf5::Group,
    df_name: &str,
    batch: &arrow::array::RecordBatch,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let schema = batch.schema();
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("dataframe"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.2.0"))?;

    // Resolve the schema field that holds the pandas index. Probe order:
    //   1. The `pandas` schema metadata's `index_columns` (the
    //      authoritative source — `pyarrow.Table.from_pandas` stamps
    //      this; covers both named and unnamed indexes).
    //   2. Fallback: `schema.field(0)` — preserves the CLI path's
    //      behaviour where obs/var come from disk without pandas
    //      metadata, and the first field already IS the index dataset.
    let pandas_idx_cols = scx_format::pandas_index_columns(schema.as_ref());
    let index_field_name: Option<String> = pandas_idx_cols
        .into_iter()
        .find(|n| schema.field_with_name(n).is_ok())
        .or_else(|| schema.fields().first().map(|f| f.name().clone()));

    // Rename pyarrow's canonical `__index_level_0__` (unnamed pandas
    // index) to anndata's `_index` literal on disk. Named indexes keep
    // their original name.
    let on_disk_index: &str = match index_field_name.as_deref() {
        Some("__index_level_0__") => "_index",
        Some(n) => n,
        None => "_index",
    };

    if !schema.fields().is_empty() {
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&vlu(on_disk_index))?;
    }

    let mut col_order: Vec<VarLenUnicode> =
        Vec::with_capacity(batch.num_columns().saturating_sub(1));
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(col_idx);
        if Some(field.name()) == index_field_name.as_ref() {
            // Index column → write under the anndata on-disk name and
            // exclude from `column-order` (matches anndata convention).
            // anndata requires `_index` to be a plain dataset, so the
            // nullable-group encoding is disabled for it.
            write_column_to_hdf5(
                group,
                df_name,
                on_disk_index,
                col,
                field.data_type(),
                false,
                sink,
            )?;
        } else if write_column_to_hdf5(
            group,
            df_name,
            field.name(),
            col,
            field.data_type(),
            true,
            sink,
        )? {
            // Only list the column in `column-order` when a dataset
            // was actually created — unsupported types are
            // warn-and-skipped and must not appear in the index.
            col_order.push(vlu(field.name()));
        }
    }

    // anndata.read_h5ad requires `column-order` to be present on every
    // dataframe group, even when the dataframe has no non-index columns
    // (it will raise `KeyError: "...can't locate attribute:
    // 'column-order'"` otherwise). Write it unconditionally — a
    // length-0 array for the no-columns case matches anndata's own
    // emission. SCX's own reader handles the empty-attr case in
    // `read_dataframe_group`'s fallback branch.
    group
        .new_attr::<VarLenUnicode>()
        .shape(col_order.len())
        .create("column-order")?
        .write_raw(&col_order)?;

    Ok(())
}

pub(super) fn write_obsm_entry_at(
    obsm_group: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
) -> Result<(), ConvertError> {
    write_obsm_entry(obsm_group, name, batch)
}

pub(super) fn write_uns_entries_at(
    group: &hdf5::Group,
    value: &serde_json::Value,
) -> Result<(), ConvertError> {
    write_uns_entries(group, value)
}

fn write_sparse_arrays(
    group: &hdf5::Group,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    // Write arrays
    group
        .new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")?
        .write(indptr)?;
    group
        .new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")?
        .write(indices)?;
    group
        .new_dataset::<f32>()
        .shape([data.len()])
        .create("data")?
        .write(data)?;

    // Set attributes
    let encoding_type = vlu("csr_matrix");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&encoding_type)?;

    let encoding_version = vlu("0.1.0");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&encoding_version)?;

    let shape = [n_obs as i64, n_vars as i64];
    group
        .new_attr::<i64>()
        .shape([2])
        .create("shape")?
        .write(&shape)?;

    Ok(())
}

fn downcast_err(name: &str, expected: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': expected {expected} array but downcast failed"
    ))
}

/// Extract codes (promoted to i32, -1 for null) and string
/// categories from a categorical column. Handles all integer key
/// widths Arrow / pandas uses (Int8/16/32/64, UInt8/16/32/64). The
/// SCX → h5ad writer emits i32 codes uniformly so anndata's reader
/// doesn't need to dispatch on key width.
fn dict_codes_and_categories_i32(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<i32>, Vec<VarLenUnicode>), ConvertError> {
    macro_rules! extract {
        ($t:ty, $label:literal) => {{
            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<$t>>()
                .ok_or_else(|| downcast_err(name, $label))?;
            let codes: Vec<i32> = dict
                .keys()
                .iter()
                .map(|v| match v {
                    Some(k) => k as i32,
                    None => -1,
                })
                .collect();
            // Categories may be narrow (`Utf8`) or wide (`LargeUtf8`);
            // dispatch on the value type like the streaming sibling
            // `dict_local_codes_and_string_values`.
            let cats: Vec<VarLenUnicode> = match dict.values().data_type() {
                DataType::Utf8 => {
                    let values_arr = dict.values().as_string::<i32>();
                    (0..values_arr.len())
                        .map(|i| vlu(values_arr.value(i)))
                        .collect()
                }
                DataType::LargeUtf8 => {
                    let values_arr = dict.values().as_string::<i64>();
                    (0..values_arr.len())
                        .map(|i| vlu(values_arr.value(i)))
                        .collect()
                }
                other => {
                    return Err(ConvertError::Other(format!(
                        "column '{name}': unsupported dictionary value type {other:?}"
                    )));
                }
            };
            Ok::<_, ConvertError>((codes, cats))
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
        DataType::Int8 => extract!(Int8Type, "Dictionary<Int8, _>"),
        DataType::Int16 => extract!(Int16Type, "Dictionary<Int16, _>"),
        DataType::Int32 => extract!(Int32Type, "Dictionary<Int32, _>"),
        DataType::Int64 => extract!(Int64Type, "Dictionary<Int64, _>"),
        DataType::UInt8 => extract!(UInt8Type, "Dictionary<UInt8, _>"),
        DataType::UInt16 => extract!(UInt16Type, "Dictionary<UInt16, _>"),
        DataType::UInt32 => extract!(UInt32Type, "Dictionary<UInt32, _>"),
        DataType::UInt64 => extract!(UInt64Type, "Dictionary<UInt64, _>"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': unsupported categorical key type {other:?}"
        ))),
    }
}

/// Collect a `Utf8` / `LargeUtf8` string column into `(values, mask)`:
/// `values[i]` is the string with null positions filled with `""`, and
/// `mask[i] == true` ⇔ row `i` is null. Dispatches on the concrete array
/// type so both narrow (`StringArray`) and wide (`LargeStringArray`)
/// offsets are handled, mirroring the streaming string writer's
/// `Utf8 | LargeUtf8` arm.
fn string_values_and_mask(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<VarLenUnicode>, Vec<bool>), ConvertError> {
    macro_rules! collect {
        ($t:ty, $label:literal) => {{
            let arr = array
                .as_any()
                .downcast_ref::<$t>()
                .ok_or_else(|| downcast_err(name, $label))?;
            let values: Vec<VarLenUnicode> = (0..arr.len())
                .map(|i| {
                    if arr.is_valid(i) {
                        vlu(arr.value(i))
                    } else {
                        vlu("")
                    }
                })
                .collect();
            let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();
            Ok((values, mask))
        }};
    }
    match array.data_type() {
        DataType::Utf8 => collect!(arrow::array::StringArray, "Utf8"),
        DataType::LargeUtf8 => collect!(LargeStringArray, "LargeUtf8"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': expected Utf8/LargeUtf8, got {other:?}"
        ))),
    }
}

/// Write one Arrow column into `group/name`. Returns `Ok(true)` on a
/// supported type (dataset created), `Ok(false)` when the type is
/// not yet supported and the column was warn-and-skipped. The caller
/// uses the boolean to decide whether to add `name` to `column-order`
/// — adding a name without a backing dataset breaks
/// `anndata.read_h5ad`'s lookup.
///
/// `allow_nullable_group` is `false` for the pandas index column
/// (`_index`), which anndata requires to be a plain dataset, never a
/// nullable group. For every other column, integer / string columns
/// that actually contain nulls are written using anndata's
/// `nullable-integer` / `nullable-string-array` group encodings
/// (`values` + `mask`), preserving null state; null-free columns stay
/// plain datasets (byte-identical to the pre-fix output). Floats are
/// always plain datasets with `NaN` at null positions — anndata has no
/// `nullable-float` encoding, so `NaN` is the canonical missing-float
/// representation.
fn write_column_to_hdf5(
    group: &hdf5::Group,
    df_name: &str,
    name: &str,
    array: &dyn Array,
    dtype: &DataType,
    allow_nullable_group: bool,
    sink: &mut WarningSink,
) -> Result<bool, ConvertError> {
    match dtype {
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(name, "Int32"))?;
            if allow_nullable_group && arr.null_count() > 0 {
                let values: Vec<i32> = (0..arr.len())
                    .map(|i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                    .collect();
                let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();
                write_nullable_group(group, name, &values, &mask, "nullable-integer")?;
            } else {
                warn_index_coerced_nulls(sink, df_name, name, "Int32", arr, allow_nullable_group);
                let values: Vec<i32> = arr.iter().map(|v| v.unwrap_or(0)).collect();
                group
                    .new_dataset::<i32>()
                    .shape([values.len()])
                    .create(name)?
                    .write(&values)?;
            }
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(name, "Int64"))?;
            if allow_nullable_group && arr.null_count() > 0 {
                let values: Vec<i64> = (0..arr.len())
                    .map(|i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                    .collect();
                let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();
                write_nullable_group(group, name, &values, &mask, "nullable-integer")?;
            } else {
                warn_index_coerced_nulls(sink, df_name, name, "Int64", arr, allow_nullable_group);
                let values: Vec<i64> = arr.iter().map(|v| v.unwrap_or(0)).collect();
                group
                    .new_dataset::<i64>()
                    .shape([values.len()])
                    .create(name)?
                    .write(&values)?;
            }
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(name, "Float32"))?;
            // NaN is anndata's canonical missing-float value — lossless,
            // unlike the prior `0.0` coercion.
            let values: Vec<f32> = arr.iter().map(|v| v.unwrap_or(f32::NAN)).collect();
            group
                .new_dataset::<f32>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(name, "Float64"))?;
            let values: Vec<f64> = arr.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            group
                .new_dataset::<f64>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            // Handle both narrow (`StringArray`, i32 offsets) and wide
            // (`LargeStringArray`, i64 offsets) string columns; the eager
            // assembled-obs path can legitimately carry either, matching
            // the streaming writer's `Utf8 | LargeUtf8` handling.
            let (values, mask) = string_values_and_mask(array, name)?;
            let null_count = mask.iter().filter(|&&m| m).count();
            if allow_nullable_group && null_count > 0 {
                write_nullable_group(group, name, &values, &mask, "nullable-string-array")?;
            } else {
                warn_index_coerced_nulls(sink, df_name, name, "Utf8", array, allow_nullable_group);
                group
                    .new_dataset::<VarLenUnicode>()
                    .shape([values.len()])
                    .create(name)?
                    .write(&values)?;
            }
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err(name, "Boolean"))?;

            // anndata's only registered IOSpec for h5py boolean
            // columns is `nullable-boolean` v0.1.0 — a *group* with
            // `values` and `mask` datasets. The legacy
            // flat-u8-with-encoding-type-boolean shape isn't
            // registered at all, so `anndata.read_h5ad` raises on
            // it. The group form additionally preserves Arrow's
            // per-element validity bits in the mask
            // (`mask[i] == 1` ⇔ row is null).
            let bool_group = group.create_group(name)?;

            // anndata + pandas's BooleanArray reader is strict: the
            // values dataset must have native HDF5 boolean dtype,
            // not u8. Plain u8 trips
            // `TypeError: values should be boolean numpy array`.
            let values: Vec<bool> = (0..arr.len())
                .map(|i| arr.is_valid(i) && arr.value(i))
                .collect();
            let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();

            bool_group
                .new_dataset::<bool>()
                .shape([values.len()])
                .create("values")?
                .write(&values)?;
            bool_group
                .new_dataset::<bool>()
                .shape([mask.len()])
                .create("mask")?
                .write(&mask)?;

            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("nullable-boolean"))?;
            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.1.0"))?;
        }
        DataType::Dictionary(_key_type, value_type)
            if matches!(value_type.as_ref(), DataType::Utf8 | DataType::LargeUtf8) =>
        {
            // Modern anndata categorical (encoding-version 0.2.0):
            // write as a group with `codes` + `categories` as
            // separate datasets, NOT as a `categories` attribute on
            // the codes dataset. The legacy attribute form overflows
            // HDF5's ~64 KB object-header limit on high-cardinality
            // categoricals (census-scale `cell_type` / `donor_id`):
            // `H5Acreate2(): object header message is too large`.
            //
            // Key types: pandas/Arrow picks the narrowest integer
            // type that fits the cardinality (Int8 for <128
            // categories, Int16 for <32K, Int32 above). We promote
            // every input to i32 on disk so the SCX → h5ad output
            // is uniform; anndata reads any width on the round-trip.
            let (codes, cats) = dict_codes_and_categories_i32(array, name)?;

            let cat_group = group.create_group(name)?;
            cat_group
                .new_dataset::<i32>()
                .shape([codes.len()])
                .create("codes")?
                .write(&codes)?;
            cat_group
                .new_dataset::<VarLenUnicode>()
                .shape([cats.len()])
                .create("categories")?
                .write(&cats)?;

            cat_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("categorical"))?;
            cat_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.2.0"))?;
            // `ordered=false` matches scipy / pandas default. Arrow's
            // DictionaryArray doesn't carry an `ordered` bit so we
            // never have richer information to forward. Native HDF5
            // bool — anndata's categorical reader expects
            // `H5T_NATIVE_HBOOL_8`, not u8.
            cat_group
                .new_attr::<bool>()
                .create("ordered")?
                .write_scalar(&false)?;
        }
        _ => {
            sink.emit(ConvertWarning::UnsupportedExportColumn {
                column: format!("{df_name}/{name}"),
                dtype: format!("{dtype:?}"),
            });
            return Ok(false);
        }
    }
    Ok(true)
}

/// Write an anndata nullable group (`encoding-type` ∈ {`nullable-integer`,
/// `nullable-string-array`}, version `0.1.0`): a subgroup with a `values`
/// dataset (null positions filled with `0` / `""`) and a boolean `mask`
/// dataset (`mask[i] == true` ⇔ null). Byte-for-byte the shape anndata's
/// `write_nullable` emits and `_read_nullable` consumes. Generic over the
/// HDF5 element type so the same helper serves integer and string values.
fn write_nullable_group<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    values: &[T],
    mask: &[bool],
    encoding_type: &str,
) -> Result<(), ConvertError> {
    let g = group.create_group(name)?;
    g.new_dataset::<T>()
        .shape([values.len()])
        .create("values")?
        .write(values)?;
    g.new_dataset::<bool>()
        .shape([mask.len()])
        .create("mask")?
        .write(mask)?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu(encoding_type))?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;
    Ok(())
}

/// Surface the one residual null-coercion case: the pandas index column
/// (`_index`), which anndata requires to be a plain dataset and so can
/// never use a nullable group. Indexes virtually never carry nulls, but
/// if one does its null becomes `0` / `""` — emit
/// [`ConvertWarning::CoercedNulls`] so that is visible. No-op for
/// non-index columns (`allow_nullable_group == true`) or when there are
/// no nulls.
fn warn_index_coerced_nulls(
    sink: &mut WarningSink,
    df_name: &str,
    col_name: &str,
    dtype: &str,
    array: &dyn Array,
    allow_nullable_group: bool,
) {
    if !allow_nullable_group && array.null_count() > 0 {
        sink.emit(ConvertWarning::CoercedNulls {
            column: format!("{df_name}/{col_name}"),
            dtype: dtype.to_string(),
            count: array.null_count() as u64,
        });
    }
}

fn write_obsm_entry(
    obsm_group: &hdf5::Group,
    name: &str,
    batch: &RecordBatch,
) -> Result<(), ConvertError> {
    let n_rows = batch.num_rows();
    let n_cols = batch.num_columns();

    // Flatten to 2D f32 array (finding 8.11: handle non-Float32 columns).
    let mut flat = vec![0.0f32; n_rows * n_cols];
    for col_idx in 0..n_cols {
        let col = batch.column(col_idx);
        if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
            for row_idx in 0..n_rows {
                flat[row_idx * n_cols + col_idx] = arr.value(row_idx);
            }
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            for row_idx in 0..n_rows {
                flat[row_idx * n_cols + col_idx] = arr.value(row_idx) as f32;
            }
        } else {
            return Err(ConvertError::Other(format!(
                "obsm '{name}' column {col_idx}: expected Float32 or Float64 array, got {:?}",
                col.data_type()
            )));
        }
    }

    let ds = obsm_group
        .new_dataset::<f32>()
        .shape([n_rows, n_cols])
        .create(name)?;

    // Write using ndarray
    let nd_array = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ConvertError::Other(format!("ndarray shape error: {e}")))?;
    ds.write(&nd_array)?;

    Ok(())
}

/// Pre-scan metadata shards to decide which integer / string columns
/// need anndata's nullable group encoding. Returns a `Vec` aligned with
/// `schema.fields()`: `true` ⇔ the field is an `Int32` / `Int64` /
/// `Utf8` / `LargeUtf8` column that actually contains ≥1 null across the
/// shards. Float and every other type always map to `false` — floats use
/// `NaN`, and bool / categorical already preserve nulls.
///
/// The streaming writer must allocate each HDF5 dataset (plain vs.
/// nullable group) before it sees any shard data, so this exact
/// null-presence signal cannot be derived from the static schema (Arrow
/// field nullability is set unconditionally by pandas → Arrow). The cost
/// is one extra decode pass over the metadata-only shards.
pub(super) fn scan_nullable_columns<I>(
    shards: I,
    schema: &Schema,
) -> Result<Vec<bool>, ConvertError>
where
    I: IntoIterator<Item = Result<RecordBatch, scx_format::error::ScxError>>,
{
    let eligible: Vec<bool> = schema
        .fields()
        .iter()
        .map(|f| {
            matches!(
                f.data_type(),
                DataType::Int32 | DataType::Int64 | DataType::Utf8 | DataType::LargeUtf8
            )
        })
        .collect();
    let mut needs = vec![false; schema.fields().len()];
    for batch_result in shards {
        let batch = batch_result?;
        for i in 0..schema.fields().len() {
            if eligible[i]
                && !needs[i]
                && i < batch.num_columns()
                && batch.column(i).null_count() > 0
            {
                needs[i] = true;
            }
        }
        // Early exit once every eligible column is already flagged.
        if eligible.iter().zip(&needs).all(|(e, n)| !e || *n) {
            break;
        }
    }
    Ok(needs)
}

/// Streaming counterpart of [`write_dataframe_group_at`]. Pre-allocates
/// one HDF5 dataset per column at fixed size `n_rows_kept`, then drains
/// `shards` and hyperslab-writes the kept-row slice of each column.
///
/// Mirrors the pre-allocate-then-hyperslab pattern in
/// `h5ad_stream_write::create_csr_triplet` + `stream_csr_to_group_at`
/// (the `/X` and `/layers/{name}` path). Peak RSS per column is bounded
/// to one shard's worth — atlas-scale obs no longer needs to live in
/// memory at once.
///
/// `schema` is taken from `ScxReader::read_obs_schema_logical_lossy()`
/// (resp. var). It carries the same `pandas` index metadata as a
/// `read_obs()` batch, so index resolution mirrors `write_dataframe_body`.
/// Per-shard batches may carry `LargeUtf8` / `Dictionary(_, LargeUtf8)`
/// even when the schema says narrow — the runtime dispatch accepts both.
///
/// `keep_mask_opt` is the global (length `n_obs`) deletion-vector keep
/// mask; `None` means no filtering. The mask is indexed by global row,
/// so this is consumed only on the obs axis (var has no DV).
///
/// `needs_nullable` is aligned with `schema.fields()`: when
/// `needs_nullable[i]` is `true`, the integer / string column at field
/// `i` is written with anndata's `nullable-integer` /
/// `nullable-string-array` group encoding (it contains nulls); otherwise
/// it is written as a plain dataset. Computed up front by
/// [`crate::h5ad_stream_write::scan_nullable_columns`] because the HDF5
/// datasets must be allocated before any shard is seen. Float columns
/// ignore this flag (always plain datasets with `NaN` at nulls).
#[allow(clippy::too_many_arguments)]
pub(super) fn write_dataframe_group_streaming<I>(
    parent: &hdf5::Group,
    name: &str,
    schema: &Schema,
    shards: I,
    n_rows_kept: usize,
    keep_mask_opt: Option<&[bool]>,
    needs_nullable: &[bool],
    sink: &mut WarningSink,
) -> Result<(), ConvertError>
where
    I: IntoIterator<Item = Result<RecordBatch, scx_format::error::ScxError>>,
{
    let group = parent.group(name).or_else(|_| parent.create_group(name))?;

    // Dataframe-level encoding attrs. Same shape as `write_dataframe_body`
    // — anndata.read_h5ad requires these even on empty obs.
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("dataframe"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.2.0"))?;

    // Resolve the pandas index field (same probe order as
    // `write_dataframe_body`: pandas metadata first, else field(0)).
    let pandas_idx_cols = scx_format::pandas_index_columns(schema);
    let index_field_name: Option<String> = pandas_idx_cols
        .into_iter()
        .find(|n| schema.field_with_name(n).is_ok())
        .or_else(|| schema.fields().first().map(|f| f.name().clone()));

    let on_disk_index: &str = match index_field_name.as_deref() {
        Some("__index_level_0__") => "_index",
        Some(n) => n,
        None => "_index",
    };

    if !schema.fields().is_empty() {
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&vlu(on_disk_index))?;
    }

    // Pre-allocate column writers and assemble `column-order` in schema
    // order (index excluded). Unsupported types are warn-and-skipped
    // inside `create_column_writer` (no dataset created) and must
    // also stay out of `column-order` so anndata's reader doesn't
    // look up a missing dataset.
    let mut col_order: Vec<VarLenUnicode> =
        Vec::with_capacity(schema.fields().len().saturating_sub(1));
    let mut col_writers: Vec<(usize, ColumnStreamWriter)> = Vec::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let is_index = Some(field.name()) == index_field_name.as_ref();
        let on_disk_name: &str = if is_index {
            on_disk_index
        } else {
            field.name()
        };
        // The index column is forced to a plain dataset (anndata requires
        // `_index` to be a plain dataset, never a nullable group).
        let want_nullable = !is_index && needs_nullable.get(col_idx).copied().unwrap_or(false);
        let writer = create_column_writer(&group, on_disk_name, field, n_rows_kept, want_nullable)?;
        if matches!(writer, ColumnStreamWriter::Unsupported) {
            sink.emit(ConvertWarning::UnsupportedExportColumn {
                column: format!("{name}/{}", field.name()),
                dtype: format!("{:?}", field.data_type()),
            });
        } else if !is_index {
            col_order.push(vlu(field.name()));
        }
        col_writers.push((col_idx, writer));
    }

    // anndata requires `column-order` even when empty (matches
    // `write_dataframe_body`).
    group
        .new_attr::<VarLenUnicode>()
        .shape(col_order.len())
        .create("column-order")?
        .write_raw(&col_order)?;

    // Drain shards. Each shard's stamped `row_start` schema metadata
    // (set by the writer via `stamp_dense_shard_metadata`) gives its
    // global row offset; the cumulative shard row count is verified
    // against it for defense-in-depth against producers that might
    // emit shards out of order.
    let mut cumulative_rows: usize = 0;
    for batch_result in shards {
        let batch = batch_result?;
        let n_shard_rows = batch.num_rows();

        // Validate the shard's schema against the declared dataframe
        // schema before touching any HDF5 dataset. A producer that
        // emits the right column *count* but reordered or retyped
        // columns would otherwise write values into the wrong
        // hyperslab — the streaming path is part of the trust boundary
        // for pipeline-generated files, so this fails loudly instead.
        validate_shard_schema(schema, &batch, name)?;

        // Prefer the stamped `row_start` over cumulative counting so
        // out-of-order producers fail loudly instead of writing into
        // wrong hyperslab offsets. Legacy shards lack the stamp and
        // fall back to `cumulative_rows` — the next cross-check is a
        // no-op for them, but v2 sharded obs always stamps it.
        let row_start_global = parse_shard_row_start(&batch).unwrap_or(cumulative_rows);
        if row_start_global != cumulative_rows {
            return Err(ConvertError::Other(format!(
                "shard '{name}' row_start {row_start_global} does not match cumulative \
                 row count {cumulative_rows} — shards must arrive in order",
            )));
        }

        // Kept-row local indices for this shard.
        let kept_local: Vec<usize> = match keep_mask_opt {
            None => (0..n_shard_rows).collect(),
            Some(mask) => {
                let upper = row_start_global + n_shard_rows;
                if mask.len() < upper {
                    return Err(ConvertError::Other(format!(
                        "keep_mask length {} < shard upper row {upper} for '{name}' \
                         (catalog/header drift)",
                        mask.len(),
                    )));
                }
                (0..n_shard_rows)
                    .filter(|&i| mask[row_start_global + i])
                    .collect()
            }
        };

        if !kept_local.is_empty() {
            for (col_idx, writer) in col_writers.iter_mut() {
                let array = batch.column(*col_idx);
                let field_name = schema.fields()[*col_idx].name();
                append_shard_to_column(writer, array, &kept_local, field_name)?;
            }
        }

        cumulative_rows += n_shard_rows;
    }

    // Validate every non-skipped column filled its pre-allocated
    // dataset exactly — a partial fill would leave default-initialised
    // trailing rows that look valid but encode incorrect data.
    for (col_idx, writer) in &col_writers {
        let field_name = schema.fields()[*col_idx].name();
        let written = column_writer_offset(writer);
        if let Some(written) = written {
            if written != n_rows_kept {
                return Err(ConvertError::Other(format!(
                    "column '{field_name}' wrote {written} rows but dataframe was \
                     pre-allocated to {n_rows_kept}",
                )));
            }
        }
    }

    // Finalize categorical writers (write `categories` + attrs).
    for (_, writer) in &col_writers {
        finalize_column_writer(writer)?;
    }

    Ok(())
}

/// Parse `row_start` from a shard's stamped schema metadata. Returns
/// `None` for legacy or non-stamped batches (callers fall back to
/// cumulative counting in that case).
fn parse_shard_row_start(batch: &RecordBatch) -> Option<usize> {
    batch
        .schema_ref()
        .metadata()
        .get("row_start")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|v| v as usize)
}

/// Number of rows written by a column streaming writer so far. Returns
/// `None` for `Unsupported` (no dataset to validate).
fn column_writer_offset(writer: &ColumnStreamWriter) -> Option<usize> {
    match writer {
        ColumnStreamWriter::Int32 { offset, .. }
        | ColumnStreamWriter::Int64 { offset, .. }
        | ColumnStreamWriter::Float32 { offset, .. }
        | ColumnStreamWriter::Float64 { offset, .. }
        | ColumnStreamWriter::Utf8 { offset, .. }
        | ColumnStreamWriter::Boolean { offset, .. }
        | ColumnStreamWriter::Categorical { offset, .. }
        | ColumnStreamWriter::Nullable { offset, .. } => Some(*offset),
        ColumnStreamWriter::Unsupported => None,
    }
}

/// Validate a per-shard `RecordBatch` against the declared dataframe
/// `schema`: same column count, same field names in the same order,
/// and logically-compatible data types. This is the streaming
/// counterpart of the assembly-time validation in
/// `assemble_sharded_metadata` — `obs_shards()` yields raw shards
/// lazily and skips that assembly, so without this a reordered or
/// retyped shard would silently land in the wrong HDF5 column.
fn validate_shard_schema(
    schema: &Schema,
    batch: &RecordBatch,
    name: &str,
) -> Result<(), ConvertError> {
    if batch.num_columns() != schema.fields().len() {
        return Err(ConvertError::Other(format!(
            "shard schema mismatch for '{name}': schema has {} fields, batch has {}",
            schema.fields().len(),
            batch.num_columns()
        )));
    }
    let batch_schema = batch.schema();
    for (i, field) in schema.fields().iter().enumerate() {
        let shard_field = batch_schema.field(i);
        if shard_field.name() != field.name() {
            return Err(ConvertError::Other(format!(
                "shard schema mismatch for '{name}': column {i} is '{}' in the dataframe \
                 schema but '{}' in the shard (columns must match by name and order)",
                field.name(),
                shard_field.name(),
            )));
        }
        if !logical_type_compatible(field.data_type(), shard_field.data_type()) {
            return Err(ConvertError::Other(format!(
                "shard schema mismatch for '{name}': column '{}' is {:?} in the dataframe \
                 schema but {:?} in the shard",
                field.name(),
                field.data_type(),
                shard_field.data_type(),
            )));
        }
    }
    Ok(())
}

/// True when a per-shard column type is interchangeable with the
/// declared schema type for the purpose of streaming export. Exact
/// equality always passes; additionally `Utf8`/`LargeUtf8` are treated
/// as equivalent and `Dictionary(_, Utf8|LargeUtf8)` are equivalent
/// regardless of key width — per-shard batches legitimately carry the
/// wide string / wide dictionary forms even when the schema says narrow
/// (the column writers dispatch on both at runtime).
fn logical_type_compatible(schema_dt: &DataType, shard_dt: &DataType) -> bool {
    fn is_string_like(t: &DataType) -> bool {
        matches!(t, DataType::Utf8 | DataType::LargeUtf8)
    }
    fn is_dict_string(t: &DataType) -> bool {
        matches!(t, DataType::Dictionary(_, v) if is_string_like(v.as_ref()))
    }
    schema_dt == shard_dt
        || (is_string_like(schema_dt) && is_string_like(shard_dt))
        || (is_dict_string(schema_dt) && is_dict_string(shard_dt))
}

/// Per-column streaming writer state. Pre-allocated HDF5 datasets +
/// running offset (for hyperslab writes) + categorical accumulator
/// (for `Dictionary(_, Utf8)` columns).
enum ColumnStreamWriter {
    Int32 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Int64 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Float32 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Float64 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Utf8 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Boolean {
        // Group is created with `encoding-type` / `encoding-version`
        // attributes up front; kept here so its lifetime extends
        // through the streaming loop (the nested datasets borrow it).
        #[allow(dead_code)]
        group: hdf5::Group,
        values_ds: hdf5::Dataset,
        mask_ds: hdf5::Dataset,
        offset: usize,
    },
    Categorical {
        group: hdf5::Group,
        codes_ds: hdf5::Dataset,
        offset: usize,
        // `dict` keys come from the per-shard dictionary values; the
        // insertion order is preserved by walking `cat_order` at finalize.
        dict: HashMap<String, i32>,
        cat_order: Vec<String>,
    },
    /// anndata `nullable-integer` / `nullable-string-array` group: a
    /// `values` dataset (nulls filled with `0` / `""`) + a boolean `mask`
    /// (`mask[i] == true` ⇔ null). Allocated only when a pre-scan found
    /// the column actually contains nulls, so null-free columns stay
    /// plain datasets. `kind` selects the Arrow downcast / value dtype.
    Nullable {
        kind: NullableKind,
        // Held so the group outlives the borrowed `values_ds`/`mask_ds`.
        #[allow(dead_code)]
        group: hdf5::Group,
        values_ds: hdf5::Dataset,
        mask_ds: hdf5::Dataset,
        offset: usize,
    },
    Unsupported,
}

/// Value dtype of a [`ColumnStreamWriter::Nullable`] column.
#[derive(Clone, Copy)]
enum NullableKind {
    Int32,
    Int64,
    String,
}

/// Pre-allocate an anndata nullable group (`values` + `mask` datasets +
/// encoding attrs) for the streaming path. Generic over the HDF5 value
/// element type so it serves both `nullable-integer` (i32/i64) and
/// `nullable-string-array` (`VarLenUnicode`). The streaming sibling of
/// [`write_nullable_group`].
fn create_nullable_writer<T: hdf5::H5Type>(
    group: &hdf5::Group,
    on_disk_name: &str,
    n_rows_kept: usize,
    encoding_type: &str,
    kind: NullableKind,
) -> Result<ColumnStreamWriter, ConvertError> {
    let g = group.create_group(on_disk_name)?;
    let values_ds = g.new_dataset::<T>().shape([n_rows_kept]).create("values")?;
    let mask_ds = g
        .new_dataset::<bool>()
        .shape([n_rows_kept])
        .create("mask")?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu(encoding_type))?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;
    Ok(ColumnStreamWriter::Nullable {
        kind,
        group: g,
        values_ds,
        mask_ds,
        offset: 0,
    })
}

fn create_column_writer(
    group: &hdf5::Group,
    on_disk_name: &str,
    field: &Field,
    n_rows_kept: usize,
    want_nullable: bool,
) -> Result<ColumnStreamWriter, ConvertError> {
    match field.data_type() {
        DataType::Int32 => {
            if want_nullable {
                return create_nullable_writer::<i32>(
                    group,
                    on_disk_name,
                    n_rows_kept,
                    "nullable-integer",
                    NullableKind::Int32,
                );
            }
            let ds = group
                .new_dataset::<i32>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Int32 { ds, offset: 0 })
        }
        DataType::Int64 => {
            if want_nullable {
                return create_nullable_writer::<i64>(
                    group,
                    on_disk_name,
                    n_rows_kept,
                    "nullable-integer",
                    NullableKind::Int64,
                );
            }
            let ds = group
                .new_dataset::<i64>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Int64 { ds, offset: 0 })
        }
        DataType::Float32 => {
            let ds = group
                .new_dataset::<f32>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Float32 { ds, offset: 0 })
        }
        DataType::Float64 => {
            let ds = group
                .new_dataset::<f64>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Float64 { ds, offset: 0 })
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            if want_nullable {
                return create_nullable_writer::<VarLenUnicode>(
                    group,
                    on_disk_name,
                    n_rows_kept,
                    "nullable-string-array",
                    NullableKind::String,
                );
            }
            let ds = group
                .new_dataset::<VarLenUnicode>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Utf8 { ds, offset: 0 })
        }
        DataType::Boolean => {
            // nullable-boolean v0.1.0: group with `values` + `mask`.
            let bool_group = group.create_group(on_disk_name)?;
            let values_ds = bool_group
                .new_dataset::<bool>()
                .shape([n_rows_kept])
                .create("values")?;
            let mask_ds = bool_group
                .new_dataset::<bool>()
                .shape([n_rows_kept])
                .create("mask")?;
            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("nullable-boolean"))?;
            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.1.0"))?;
            Ok(ColumnStreamWriter::Boolean {
                group: bool_group,
                values_ds,
                mask_ds,
                offset: 0,
            })
        }
        DataType::Dictionary(_, value_type)
            if matches!(value_type.as_ref(), DataType::Utf8 | DataType::LargeUtf8) =>
        {
            let cat_group = group.create_group(on_disk_name)?;
            let codes_ds = cat_group
                .new_dataset::<i32>()
                .shape([n_rows_kept])
                .create("codes")?;
            Ok(ColumnStreamWriter::Categorical {
                group: cat_group,
                codes_ds,
                offset: 0,
                dict: HashMap::new(),
                cat_order: Vec::new(),
            })
        }
        _ => {
            // Mirrors `write_column_to_hdf5`'s warn-and-skip arm so the
            // streaming and eager paths behave identically on
            // unsupported types. The `UnsupportedExportColumn` warning
            // is emitted by the caller (which holds the `WarningSink`).
            Ok(ColumnStreamWriter::Unsupported)
        }
    }
}

fn append_shard_to_column(
    writer: &mut ColumnStreamWriter,
    array: &dyn Array,
    kept_local: &[usize],
    name: &str,
) -> Result<(), ConvertError> {
    match writer {
        ColumnStreamWriter::Int32 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(name, "Int32"))?;
            let values: Vec<i32> = kept_local
                .iter()
                .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Int64 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(name, "Int64"))?;
            let values: Vec<i64> = kept_local
                .iter()
                .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Float32 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(name, "Float32"))?;
            let values: Vec<f32> = kept_local
                .iter()
                .map(|&i| {
                    if arr.is_valid(i) {
                        arr.value(i)
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Float64 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(name, "Float64"))?;
            let values: Vec<f64> = kept_local
                .iter()
                .map(|&i| {
                    if arr.is_valid(i) {
                        arr.value(i)
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Utf8 { ds, offset } => {
            let values: Vec<VarLenUnicode> = match array.data_type() {
                DataType::Utf8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| downcast_err(name, "Utf8"))?;
                    kept_local
                        .iter()
                        .map(|&i| {
                            if arr.is_valid(i) {
                                vlu(arr.value(i))
                            } else {
                                vlu("")
                            }
                        })
                        .collect()
                }
                DataType::LargeUtf8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or_else(|| downcast_err(name, "LargeUtf8"))?;
                    kept_local
                        .iter()
                        .map(|&i| {
                            if arr.is_valid(i) {
                                vlu(arr.value(i))
                            } else {
                                vlu("")
                            }
                        })
                        .collect()
                }
                other => {
                    return Err(ConvertError::Other(format!(
                        "column '{name}': expected Utf8/LargeUtf8, got {other:?}"
                    )));
                }
            };
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Boolean {
            values_ds,
            mask_ds,
            offset,
            ..
        } => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err(name, "Boolean"))?;
            let values: Vec<bool> = kept_local
                .iter()
                .map(|&i| arr.is_valid(i) && arr.value(i))
                .collect();
            let mask: Vec<bool> = kept_local.iter().map(|&i| !arr.is_valid(i)).collect();
            values_ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            mask_ds.write_slice(
                ArrayView1::from(mask.as_slice()),
                ndarray::s![*offset..*offset + mask.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Categorical {
            codes_ds,
            offset,
            dict,
            cat_order,
            ..
        } => {
            let (local_codes, local_values) = dict_local_codes_and_string_values(array, name)?;
            // C10: intern only the dictionary values actually referenced by
            // `kept_local` rows, so categories present only in
            // deletion-dropped rows don't enter the global vocabulary. The
            // local→global map is filled lazily as kept codes are visited.
            let mut local_to_global: Vec<Option<i32>> = vec![None; local_values.len()];
            let mut kept_codes: Vec<i32> = Vec::with_capacity(kept_local.len());
            for &i in kept_local {
                let lc = local_codes[i];
                if lc < 0 {
                    kept_codes.push(-1);
                    continue;
                }
                let lc_idx = lc as usize;
                let g = match local_to_global[lc_idx] {
                    Some(g) => g,
                    None => {
                        let v = &local_values[lc_idx];
                        let g = match dict.get(v) {
                            Some(&g) => g,
                            None => {
                                // Cap at i32::MAX. Practical categorical
                                // cardinalities (cell_type, donor_id) stay
                                // well below this; saturating is defensive.
                                let g: i32 = cat_order.len().try_into().map_err(|_| {
                                    ConvertError::Other(format!(
                                        "column '{name}': categorical cardinality exceeds i32::MAX"
                                    ))
                                })?;
                                dict.insert(v.clone(), g);
                                cat_order.push(v.clone());
                                g
                            }
                        };
                        local_to_global[lc_idx] = Some(g);
                        g
                    }
                };
                kept_codes.push(g);
            }
            codes_ds.write_slice(
                ArrayView1::from(kept_codes.as_slice()),
                ndarray::s![*offset..*offset + kept_codes.len()],
            )?;
            *offset += kept_codes.len();
        }
        ColumnStreamWriter::Nullable {
            kind,
            values_ds,
            mask_ds,
            offset,
            ..
        } => {
            let mask: Vec<bool> = kept_local.iter().map(|&i| !array.is_valid(i)).collect();
            match kind {
                NullableKind::Int32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .ok_or_else(|| downcast_err(name, "Int32"))?;
                    let values: Vec<i32> = kept_local
                        .iter()
                        .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                        .collect();
                    values_ds.write_slice(
                        ArrayView1::from(values.as_slice()),
                        ndarray::s![*offset..*offset + values.len()],
                    )?;
                }
                NullableKind::Int64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| downcast_err(name, "Int64"))?;
                    let values: Vec<i64> = kept_local
                        .iter()
                        .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                        .collect();
                    values_ds.write_slice(
                        ArrayView1::from(values.as_slice()),
                        ndarray::s![*offset..*offset + values.len()],
                    )?;
                }
                NullableKind::String => {
                    // Per-shard arrays may be Utf8 or LargeUtf8 (mirrors the
                    // plain `Utf8` writer arm).
                    let values: Vec<VarLenUnicode> = match array.data_type() {
                        DataType::Utf8 => {
                            let arr = array
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .ok_or_else(|| downcast_err(name, "Utf8"))?;
                            kept_local
                                .iter()
                                .map(|&i| {
                                    if arr.is_valid(i) {
                                        vlu(arr.value(i))
                                    } else {
                                        vlu("")
                                    }
                                })
                                .collect()
                        }
                        DataType::LargeUtf8 => {
                            let arr = array
                                .as_any()
                                .downcast_ref::<LargeStringArray>()
                                .ok_or_else(|| downcast_err(name, "LargeUtf8"))?;
                            kept_local
                                .iter()
                                .map(|&i| {
                                    if arr.is_valid(i) {
                                        vlu(arr.value(i))
                                    } else {
                                        vlu("")
                                    }
                                })
                                .collect()
                        }
                        other => {
                            return Err(ConvertError::Other(format!(
                                "column '{name}': expected Utf8/LargeUtf8, got {other:?}"
                            )));
                        }
                    };
                    values_ds.write_slice(
                        ArrayView1::from(values.as_slice()),
                        ndarray::s![*offset..*offset + values.len()],
                    )?;
                }
            }
            mask_ds.write_slice(
                ArrayView1::from(mask.as_slice()),
                ndarray::s![*offset..*offset + mask.len()],
            )?;
            *offset += mask.len();
        }
        ColumnStreamWriter::Unsupported => {
            // Warning emitted once at writer creation; per-shard
            // append is a no-op (mirrors the eager path's skip).
        }
    }
    Ok(())
}

fn finalize_column_writer(writer: &ColumnStreamWriter) -> Result<(), ConvertError> {
    if let ColumnStreamWriter::Categorical {
        group, cat_order, ..
    } = writer
    {
        let cats: Vec<VarLenUnicode> = cat_order.iter().map(|s| vlu(s)).collect();
        group
            .new_dataset::<VarLenUnicode>()
            .shape([cats.len()])
            .create("categories")?
            .write(&cats)?;
        group
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")?
            .write_scalar(&vlu("categorical"))?;
        group
            .new_attr::<VarLenUnicode>()
            .create("encoding-version")?
            .write_scalar(&vlu("0.2.0"))?;
        group
            .new_attr::<bool>()
            .create("ordered")?
            .write_scalar(&false)?;
    }
    Ok(())
}

/// Streaming counterpart of [`dict_codes_and_categories_i32`]: extracts
/// local codes (promoted to i32, -1 for null) and the local dictionary
/// values as owned `Vec<String>` so callers can fold values into a
/// running global dictionary across shards. Accepts both
/// `Dictionary(_, Utf8)` and `Dictionary(_, LargeUtf8)` value types so
/// shard runtime arrays match whatever the per-shard downcast left.
fn dict_local_codes_and_string_values(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<i32>, Vec<String>), ConvertError> {
    macro_rules! extract {
        ($t:ty, $label:literal) => {{
            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<$t>>()
                .ok_or_else(|| downcast_err(name, $label))?;
            let codes: Vec<i32> = dict
                .keys()
                .iter()
                .map(|v| match v {
                    Some(k) => k as i32,
                    None => -1,
                })
                .collect();
            let values: Vec<String> = match dict.values().data_type() {
                DataType::Utf8 => {
                    let arr = dict.values().as_string::<i32>();
                    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
                }
                DataType::LargeUtf8 => {
                    let arr = dict.values().as_string::<i64>();
                    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
                }
                other => {
                    return Err(ConvertError::Other(format!(
                        "column '{name}': unsupported dictionary value type {other:?}"
                    )));
                }
            };
            Ok::<_, ConvertError>((codes, values))
        }};
    }
    let key_type = match array.data_type() {
        DataType::Dictionary(k, _) => k.as_ref(),
        _ => {
            return Err(ConvertError::Other(format!(
                "column '{name}': expected Dictionary, got {:?}",
                array.data_type()
            )));
        }
    };
    match key_type {
        DataType::Int8 => extract!(Int8Type, "Dictionary<Int8, _>"),
        DataType::Int16 => extract!(Int16Type, "Dictionary<Int16, _>"),
        DataType::Int32 => extract!(Int32Type, "Dictionary<Int32, _>"),
        DataType::Int64 => extract!(Int64Type, "Dictionary<Int64, _>"),
        DataType::UInt8 => extract!(UInt8Type, "Dictionary<UInt8, _>"),
        DataType::UInt16 => extract!(UInt16Type, "Dictionary<UInt16, _>"),
        DataType::UInt32 => extract!(UInt32Type, "Dictionary<UInt32, _>"),
        DataType::UInt64 => extract!(UInt64Type, "Dictionary<UInt64, _>"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': unsupported categorical key type {other:?}"
        ))),
    }
}

fn write_uns_entries(group: &hdf5::Group, value: &serde_json::Value) -> Result<(), ConvertError> {
    if let serde_json::Value::Object(map) = value {
        for (key, val) in map {
            write_uns_value(group, key, val)?;
        }
    }
    Ok(())
}

fn write_uns_value(
    group: &hdf5::Group,
    name: &str,
    value: &serde_json::Value,
) -> Result<(), ConvertError> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                group
                    .new_dataset::<i64>()
                    .shape(())
                    .create(name)?
                    .write_scalar(&i)?;
            } else if let Some(f) = n.as_f64() {
                group
                    .new_dataset::<f64>()
                    .shape(())
                    .create(name)?
                    .write_scalar(&f)?;
            }
        }
        serde_json::Value::String(s) => {
            let v = vlu(s);
            group
                .new_dataset::<VarLenUnicode>()
                .shape(())
                .create(name)?
                .write_scalar(&v)?;
        }
        serde_json::Value::Bool(b) => {
            group
                .new_dataset::<bool>()
                .shape(())
                .create(name)?
                .write_scalar(b)?;
        }
        serde_json::Value::Array(arr) => {
            // 2-D numeric arrays (e.g. color/contrast matrices) — mirror the
            // nested-JSON shape produced by `read_uns_entry`'s 2-D arm so they
            // round-trip rather than being silently dropped. Ragged / empty /
            // non-numeric nested arrays fall through to the no-op skip (a true
            // HDF5 round-trip is always rectangular, so ragged only arises from
            // synthetic JSON).
            if !arr.is_empty() && arr.iter().all(|v| v.is_array()) {
                write_uns_2d_array(group, name, arr)?;
            } else if arr.iter().all(|v| v.is_i64()) {
                let data: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                group
                    .new_dataset::<i64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if !arr.is_empty() && arr.iter().all(|v| v.is_u64()) {
                // Unsigned values above i64::MAX (C3): keep them exact instead
                // of coercing to f64 via the `is_number` arm below.
                let data: Vec<u64> = arr.iter().filter_map(|v| v.as_u64()).collect();
                group
                    .new_dataset::<u64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if arr.iter().all(|v| v.is_number()) {
                let data: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                group
                    .new_dataset::<f64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if !arr.is_empty() && arr.iter().all(|v| v.is_boolean()) {
                // 1-D boolean (C3): the scalar bool arm above confirms the
                // `bool` H5Type; without this arm bool vectors are dropped.
                let data: Vec<bool> = arr.iter().filter_map(|v| v.as_bool()).collect();
                group
                    .new_dataset::<bool>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if arr.iter().all(|v| v.is_string()) {
                let data: Vec<VarLenUnicode> =
                    arr.iter().filter_map(|v| v.as_str()).map(vlu).collect();
                group
                    .new_dataset::<VarLenUnicode>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            }
        }
        serde_json::Value::Object(_) => {
            let subgroup = group.create_group(name)?;
            write_uns_entries(&subgroup, value)?;
        }
        serde_json::Value::Null => {}
    }
    Ok(())
}

/// Write a 2-D numeric `uns` array (nested JSON `[[..],[..]]`) as a rectangular
/// HDF5 dataset, mirroring the dtype set of `read_uns_entry`'s 2-D arm
/// (Integer/Unsigned/Float/Boolean). Ragged, empty, or non-numeric inputs are
/// skipped (no-op) rather than errored, matching the lenient behavior of the
/// surrounding scalar/1-D arms. The caller guarantees every element is an array.
fn write_uns_2d_array(
    group: &hdf5::Group,
    name: &str,
    arr: &[serde_json::Value],
) -> Result<(), ConvertError> {
    let rows: Vec<&Vec<serde_json::Value>> = arr.iter().filter_map(|v| v.as_array()).collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    // Rectangular and non-degenerate, else skip.
    if n_cols == 0 || rows.iter().any(|r| r.len() != n_cols) {
        return Ok(());
    }
    let cells = || rows.iter().flat_map(|r| r.iter());

    // Detect element dtype over all cells, mirroring the read-path ordering:
    // i64 first (catches negatives + small ints), then u64 (large unsigned),
    // then f64, then bool. Anything else (mixed / non-numeric) is skipped.
    if cells().all(|v| v.is_i64()) {
        let flat: Vec<i64> = cells().filter_map(|v| v.as_i64()).collect();
        write_2d_dataset::<i64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_u64()) {
        let flat: Vec<u64> = cells().filter_map(|v| v.as_u64()).collect();
        write_2d_dataset::<u64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_number()) {
        let flat: Vec<f64> = cells().filter_map(|v| v.as_f64()).collect();
        write_2d_dataset::<f64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_boolean()) {
        let flat: Vec<bool> = cells().filter_map(|v| v.as_bool()).collect();
        write_2d_dataset::<bool>(group, name, n_rows, n_cols, flat)
    } else {
        Ok(())
    }
}

/// Create and write a rectangular `n_rows × n_cols` HDF5 dataset from a
/// row-major flattened buffer. Shared by the `write_uns_2d_array` type arms;
/// mirrors the `ndarray::Array2::from_shape_vec` idiom in `write_obsm_entry`.
fn write_2d_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    flat: Vec<T>,
) -> Result<(), ConvertError> {
    let nd = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ConvertError::Other(format!("ndarray shape error: {e}")))?;
    group
        .new_dataset::<T>()
        .shape([n_rows, n_cols])
        .create(name)?
        .write(&nd)?;
    Ok(())
}
