// Write scx file back to h5ad format

use std::path::Path;

use arrow::array::{
    Array, AsArray, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch,
};
use arrow::datatypes::{DataType, Int32Type};
use hdf5::types::VarLenUnicode;

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

/// Write an SCX file to h5ad format.
pub fn write_scx_to_h5ad(scx_path: &Path, h5ad_path: &Path) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    let file = hdf5::File::create(h5ad_path)?;

    // Read the full CSR matrix
    let csr = reader.read_all_csr_shards()?;
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

    // Write obs
    if let Ok(obs) = reader.read_obs() {
        write_dataframe_group(&file, "obs", &obs)?;
    }

    // Write var
    if let Ok(var) = reader.read_var() {
        write_dataframe_group(&file, "var", &var)?;
    }

    // Write obsm
    if let Ok(obsm_map) = reader.read_all_obsm() {
        if !obsm_map.is_empty() {
            let obsm_group = file.create_group("obsm")?;
            for (name, batch) in &obsm_map {
                write_obsm_entry(&obsm_group, name, batch)?;
            }
        }
    }

    // Write uns
    if let Ok(uns) = reader.read_uns() {
        let uns_group = file.create_group("uns")?;
        write_uns_entries(&uns_group, &uns)?;
    }

    // Write layers
    let layer_names = reader.layer_names();
    if !layer_names.is_empty() {
        let layers_group = file.create_group("layers")?;
        for layer_name in &layer_names {
            if let Ok(layer_csr) = reader.read_layer(layer_name) {
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

    Ok(())
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
) -> Result<(), ConvertError> {
    let group = parent.group(name).or_else(|_| parent.create_group(name))?;
    let schema = batch.schema();
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("dataframe"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.2.0"))?;

    // `_index` attribute names the column that holds the row index.
    // anndata.read_h5ad requires this; without it the `/obs` group
    // fails to read.
    if !schema.fields().is_empty() {
        let index_name = vlu(schema.field(0).name());
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&index_name)?;
    }

    let mut col_order: Vec<VarLenUnicode> = Vec::with_capacity(batch.num_columns());
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(col_idx);
        write_column_to_hdf5(&group, field.name(), col, field.data_type())?;
        col_order.push(vlu(field.name()));
    }

    if !col_order.is_empty() {
        group
            .new_attr::<VarLenUnicode>()
            .shape(col_order.len())
            .create("column-order")?
            .write_raw(&col_order)?;
    }

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

fn write_dataframe_group(
    file: &hdf5::File,
    name: &str,
    batch: &RecordBatch,
) -> Result<(), ConvertError> {
    let group = file.create_group(name)?;

    let schema = batch.schema();

    // Write _index attribute (use first column name)
    if !schema.fields().is_empty() {
        let index_name = vlu(schema.field(0).name());
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&index_name)?;
    }

    // Write column-categories group attribute
    let encoding_type = vlu("dataframe");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&encoding_type)?;

    let encoding_version = vlu("0.2.0");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&encoding_version)?;

    // Write column-order attribute
    let col_names: Vec<VarLenUnicode> = schema.fields().iter().map(|f| vlu(f.name())).collect();
    group
        .new_attr::<VarLenUnicode>()
        .shape([col_names.len()])
        .create("column-order")?
        .write(&col_names)?;

    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        write_column_to_hdf5(&group, field.name(), col, field.data_type())?;
    }

    Ok(())
}

fn downcast_err(name: &str, expected: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': expected {expected} array but downcast failed"
    ))
}

fn write_column_to_hdf5(
    group: &hdf5::Group,
    name: &str,
    array: &dyn Array,
    dtype: &DataType,
) -> Result<(), ConvertError> {
    match dtype {
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(name, "Int32"))?;
            let values: Vec<i32> = arr.iter().map(|v| v.unwrap_or(0)).collect();
            group
                .new_dataset::<i32>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(name, "Int64"))?;
            let values: Vec<i64> = arr.iter().map(|v| v.unwrap_or(0)).collect();
            group
                .new_dataset::<i64>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(name, "Float32"))?;
            let values: Vec<f32> = arr.iter().map(|v| v.unwrap_or(0.0)).collect();
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
            let values: Vec<f64> = arr.iter().map(|v| v.unwrap_or(0.0)).collect();
            group
                .new_dataset::<f64>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| downcast_err(name, "Utf8"))?;
            let values: Vec<VarLenUnicode> = arr.iter().map(|v| vlu(v.unwrap_or(""))).collect();
            group
                .new_dataset::<VarLenUnicode>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err(name, "Boolean"))?;
            let values: Vec<u8> = arr
                .iter()
                .map(|v| if v.unwrap_or(false) { 1u8 } else { 0u8 })
                .collect();
            let ds = group
                .new_dataset::<u8>()
                .shape([values.len()])
                .create(name)?;
            ds.write(&values)?;
            let enc = vlu("boolean");
            ds.new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&enc)?;
        }
        DataType::Dictionary(key_type, value_type)
            if **key_type == DataType::Int32 && **value_type == DataType::Utf8 =>
        {
            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| downcast_err(name, "Dictionary<Int32, Utf8>"))?;

            // Write codes
            let keys = dict.keys();
            let codes: Vec<i32> = keys.iter().map(|v| v.unwrap_or(-1)).collect();
            let ds = group
                .new_dataset::<i32>()
                .shape([codes.len()])
                .create(name)?;
            ds.write(&codes)?;

            // Set encoding-type
            let enc = vlu("categorical");
            ds.new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&enc)?;

            // Write categories as attribute
            let values_arr = dict.values().as_string::<i32>();
            let cats: Vec<VarLenUnicode> = (0..values_arr.len())
                .map(|i| vlu(values_arr.value(i)))
                .collect();
            ds.new_attr::<VarLenUnicode>()
                .shape([cats.len()])
                .create("categories")?
                .write(&cats)?;
        }
        _ => {
            eprintln!("warning: skipping column '{name}' with unsupported type {dtype:?}");
        }
    }
    Ok(())
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
            let v = if *b { 1u8 } else { 0u8 };
            group
                .new_dataset::<u8>()
                .shape(())
                .create(name)?
                .write_scalar(&v)?;
        }
        serde_json::Value::Array(arr) => {
            // Try as array of numbers
            if arr.iter().all(|v| v.is_i64()) {
                let data: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                group
                    .new_dataset::<i64>()
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
