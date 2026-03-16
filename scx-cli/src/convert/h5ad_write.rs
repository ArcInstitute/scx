// Write scx file back to h5ad format

use std::path::Path;

use arrow::array::{
    Array, AsArray, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch,
};
use arrow::datatypes::{DataType, Int32Type};
use hdf5::types::VarLenUnicode;

/// Convert &str to VarLenUnicode (infallible for non-NUL strings).
fn vlu(s: &str) -> VarLenUnicode {
    // Safety: our strings never contain NUL bytes
    unsafe { VarLenUnicode::from_str_unchecked(s) }
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

fn write_column_to_hdf5(
    group: &hdf5::Group,
    name: &str,
    array: &dyn Array,
    dtype: &DataType,
) -> Result<(), ConvertError> {
    match dtype {
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let values: Vec<i32> = arr.iter().map(|v| v.unwrap_or(0)).collect();
            group
                .new_dataset::<i32>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let values: Vec<i64> = arr.iter().map(|v| v.unwrap_or(0)).collect();
            group
                .new_dataset::<i64>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let values: Vec<f32> = arr.iter().map(|v| v.unwrap_or(0.0)).collect();
            group
                .new_dataset::<f32>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
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
                .unwrap();
            let values: Vec<VarLenUnicode> = arr.iter().map(|v| vlu(v.unwrap_or(""))).collect();
            group
                .new_dataset::<VarLenUnicode>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
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
                .unwrap();

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

    // Flatten to 2D f32 array
    let mut flat = vec![0.0f32; n_rows * n_cols];
    for col_idx in 0..n_cols {
        let arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for row_idx in 0..n_rows {
            flat[row_idx * n_cols + col_idx] = arr.value(row_idx);
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
