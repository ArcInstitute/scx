// Read h5ad file components: X matrix, obs, var, obsm, uns, layers

use std::collections::HashMap;

use arrow::array::{
    ArrayRef, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use hdf5::types::TypeDescriptor;
use std::sync::Arc;

use super::csc_transpose::csc_to_csr;
use super::detect::MatrixFormat;
use super::pipeline::ConvertError;

/// CSR matrix arrays + shape: (indptr, indices, data, n_obs, n_vars)
type CsrArrays = (Vec<i64>, Vec<i32>, Vec<f32>, usize, usize);

/// Read the X matrix from an h5ad file, returning CSR arrays and shape.
pub fn read_x_matrix(file: &hdf5::File, format: MatrixFormat) -> Result<CsrArrays, ConvertError> {
    match format {
        MatrixFormat::Csr => read_sparse_matrix(file, "X", false),
        MatrixFormat::Csc => read_sparse_matrix(file, "X", true),
        MatrixFormat::Dense => read_dense_matrix(file, "X"),
    }
}

/// Read a sparse matrix group (CSR or CSC) and return as CSR.
fn read_sparse_matrix(
    file: &hdf5::File,
    group_name: &str,
    is_csc: bool,
) -> Result<CsrArrays, ConvertError> {
    let group = file.group(group_name)?;

    // Read shape from group attribute
    let shape: Vec<i64> = group.attr("shape")?.read_1d()?.to_vec();
    let (n_obs, n_vars) = (shape[0] as usize, shape[1] as usize);

    // Read indptr with runtime dtype handling
    let indptr_ds = group.dataset("indptr")?;
    let indptr = read_i64_dataset(&indptr_ds)?;

    // Read indices
    let indices_ds = group.dataset("indices")?;
    let indices = read_i32_dataset(&indices_ds)?;

    // Read data
    let data_ds = group.dataset("data")?;
    let data = read_f32_dataset(&data_ds)?;

    if is_csc {
        // CSC shape is [n_obs, n_vars] but indptr length = n_vars + 1
        let (csr_indptr, csr_indices, csr_data) =
            csc_to_csr(&indptr, &indices, &data, n_obs, n_vars);
        Ok((csr_indptr, csr_indices, csr_data, n_obs, n_vars))
    } else {
        Ok((indptr, indices, data, n_obs, n_vars))
    }
}

/// Read a dense X dataset and convert to CSR.
fn read_dense_matrix(file: &hdf5::File, dataset_name: &str) -> Result<CsrArrays, ConvertError> {
    let ds = file.dataset(dataset_name)?;
    let shape = ds.shape();
    let n_obs = shape[0];
    let n_vars = shape[1];

    // Read 2D dataset as ndarray then flatten to Vec<f32>
    let nd: ndarray::Array2<f32> = ds.read_2d()?;
    let flat: Vec<f32> = nd.into_raw_vec();
    let csr = scx_sparse::dense_to_csr(&flat, n_obs, n_vars)
        .map_err(|e| ConvertError::Other(format!("CSR conversion error: {e}")))?;

    Ok((csr.indptr, csr.indices, csr.data, n_obs, n_vars))
}

/// Read a dataset as Vec<i64>, handling i32 or i64 source dtypes.
pub(super) fn read_i64_dataset(ds: &hdf5::Dataset) -> Result<Vec<i64>, ConvertError> {
    let desc = ds.dtype()?.to_descriptor()?;
    match desc {
        TypeDescriptor::Integer(sz) => match sz {
            hdf5::types::IntSize::U4 => {
                let data: Vec<i32> = ds.read_1d()?.to_vec();
                Ok(data.iter().map(|&v| v as i64).collect())
            }
            hdf5::types::IntSize::U8 => {
                let data: Vec<i64> = ds.read_1d()?.to_vec();
                Ok(data)
            }
            _ => {
                // Try reading as i64 for other sizes
                let data: Vec<i64> = ds.read_1d()?.to_vec();
                Ok(data)
            }
        },
        TypeDescriptor::Unsigned(sz) => match sz {
            hdf5::types::IntSize::U4 => {
                let data: Vec<u32> = ds.read_1d()?.to_vec();
                Ok(data.iter().map(|&v| v as i64).collect())
            }
            hdf5::types::IntSize::U8 => {
                let data: Vec<u64> = ds.read_1d()?.to_vec();
                Ok(data.iter().map(|&v| v as i64).collect())
            }
            _ => {
                let data: Vec<i64> = ds.read_1d()?.to_vec();
                Ok(data)
            }
        },
        _ => {
            // Fallback
            let data: Vec<i64> = ds.read_1d()?.to_vec();
            Ok(data)
        }
    }
}

/// Read a dataset as Vec<i32>, handling various integer dtypes.
pub(super) fn read_i32_dataset(ds: &hdf5::Dataset) -> Result<Vec<i32>, ConvertError> {
    let desc = ds.dtype()?.to_descriptor()?;
    match desc {
        TypeDescriptor::Integer(sz) => match sz {
            hdf5::types::IntSize::U4 => {
                let data: Vec<i32> = ds.read_1d()?.to_vec();
                Ok(data)
            }
            hdf5::types::IntSize::U8 => {
                let data: Vec<i64> = ds.read_1d()?.to_vec();
                Ok(data.iter().map(|&v| v as i32).collect())
            }
            _ => {
                let data: Vec<i32> = ds.read_1d()?.to_vec();
                Ok(data)
            }
        },
        TypeDescriptor::Unsigned(hdf5::types::IntSize::U4) => {
            let data: Vec<u32> = ds.read_1d()?.to_vec();
            Ok(data.iter().map(|&v| v as i32).collect())
        }
        _ => {
            let data: Vec<i32> = ds.read_1d()?.to_vec();
            Ok(data)
        }
    }
}

/// Read a dataset as Vec<f32>, handling f32, f64, i32, u32 source dtypes.
pub(super) fn read_f32_dataset(ds: &hdf5::Dataset) -> Result<Vec<f32>, ConvertError> {
    let desc = ds.dtype()?.to_descriptor()?;
    match desc {
        TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
            let data: Vec<f64> = ds.read_1d()?.to_vec();
            Ok(data.iter().map(|&v| v as f32).collect())
        }
        TypeDescriptor::Integer(hdf5::types::IntSize::U4) => {
            let data: Vec<i32> = ds.read_1d()?.to_vec();
            Ok(data.iter().map(|&v| v as f32).collect())
        }
        TypeDescriptor::Unsigned(hdf5::types::IntSize::U4) => {
            let data: Vec<u32> = ds.read_1d()?.to_vec();
            Ok(data.iter().map(|&v| v as f32).collect())
        }
        _ => {
            // Fallback: try reading as f32 (HDF5 may auto-convert)
            let data: Vec<f32> = ds.read_1d()?.to_vec();
            Ok(data)
        }
    }
}

/// Read a DataFrame group (obs or var) from an h5ad file as an Arrow RecordBatch.
pub fn read_dataframe_group(
    file: &hdf5::File,
    group_name: &str,
) -> Result<RecordBatch, ConvertError> {
    let group = file.group(group_name)?;

    // Get the index column name from the _index attribute
    let index_col_name: Option<String> = group
        .attr("_index")
        .ok()
        .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.to_string());

    let member_names = group.member_names()?;

    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for name in &member_names {
        // Skip internal members
        if name.starts_with("__") {
            continue;
        }

        // Try reading as dataset
        let ds = match group.dataset(name) {
            Ok(ds) => ds,
            Err(_) => continue, // Skip non-dataset members (subgroups)
        };

        match read_column_to_arrow(&group, &ds, name) {
            Ok((field, array)) => {
                fields.push(field);
                arrays.push(array);
            }
            Err(e) => {
                eprintln!("warning: skipping column '{name}' in {group_name}: {e}");
            }
        }
    }

    if fields.is_empty() {
        // Create an empty batch with a dummy index if needed
        let n_rows = if let Some(ref idx_name) = index_col_name {
            if let Ok(ds) = group.dataset(idx_name) {
                ds.shape()[0]
            } else {
                0
            }
        } else {
            0
        };
        let schema = Schema::new(vec![Field::new("_index", DataType::Utf8, true)]);
        let empty_arr: ArrayRef = Arc::new(StringArray::from(vec![""; n_rows]));
        return Ok(RecordBatch::try_new(Arc::new(schema), vec![empty_arr])?);
    }

    let schema = Schema::new(fields);
    Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
}

/// Read a single column dataset to an Arrow array.
fn read_column_to_arrow(
    group: &hdf5::Group,
    ds: &hdf5::Dataset,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    // Check for categorical encoding
    let is_categorical = ds
        .attr("encoding-type")
        .ok()
        .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.as_str() == "categorical")
        .unwrap_or(false);

    if is_categorical {
        return read_categorical_column(group, ds, name);
    }

    let desc = ds.dtype()?.to_descriptor()?;
    match desc {
        TypeDescriptor::Integer(sz) => match sz {
            hdf5::types::IntSize::U1 => {
                // Could be boolean encoding
                let data: Vec<i8> = ds.read_1d()?.to_vec();
                let array: ArrayRef = Arc::new(Int32Array::from(
                    data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                ));
                Ok((Field::new(name, DataType::Int32, true), array))
            }
            hdf5::types::IntSize::U2 => {
                let data: Vec<i16> = ds.read_1d()?.to_vec();
                let array: ArrayRef = Arc::new(Int32Array::from(
                    data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                ));
                Ok((Field::new(name, DataType::Int32, true), array))
            }
            hdf5::types::IntSize::U4 => {
                let data: Vec<i32> = ds.read_1d()?.to_vec();
                let array: ArrayRef = Arc::new(Int32Array::from(data));
                Ok((Field::new(name, DataType::Int32, true), array))
            }
            hdf5::types::IntSize::U8 => {
                let data: Vec<i64> = ds.read_1d()?.to_vec();
                let array: ArrayRef = Arc::new(Int64Array::from(data));
                Ok((Field::new(name, DataType::Int64, true), array))
            }
        },
        TypeDescriptor::Unsigned(sz) => match sz {
            hdf5::types::IntSize::U1 => {
                // Check if this is actually a boolean
                let is_bool = ds
                    .attr("encoding-type")
                    .ok()
                    .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
                    .map(|v| v.as_str() == "boolean")
                    .unwrap_or(false);
                if is_bool {
                    let data: Vec<u8> = ds.read_1d()?.to_vec();
                    let array: ArrayRef = Arc::new(BooleanArray::from(
                        data.iter().map(|&v| v != 0).collect::<Vec<_>>(),
                    ));
                    Ok((Field::new(name, DataType::Boolean, true), array))
                } else {
                    let data: Vec<u8> = ds.read_1d()?.to_vec();
                    let array: ArrayRef = Arc::new(Int32Array::from(
                        data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                    ));
                    Ok((Field::new(name, DataType::Int32, true), array))
                }
            }
            hdf5::types::IntSize::U4 => {
                let data: Vec<u32> = ds.read_1d()?.to_vec();
                let array: ArrayRef = Arc::new(Int64Array::from(
                    data.iter().map(|&v| v as i64).collect::<Vec<_>>(),
                ));
                Ok((Field::new(name, DataType::Int64, true), array))
            }
            _ => {
                let data: Vec<i32> = ds.read_1d()?.to_vec();
                let array: ArrayRef = Arc::new(Int32Array::from(data));
                Ok((Field::new(name, DataType::Int32, true), array))
            }
        },
        TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
            let data: Vec<f64> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(Float64Array::from(data));
            Ok((Field::new(name, DataType::Float64, true), array))
        }
        TypeDescriptor::Float(_) => {
            let data: Vec<f32> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(Float32Array::from(data));
            Ok((Field::new(name, DataType::Float32, true), array))
        }
        TypeDescriptor::Boolean => {
            let data: Vec<bool> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(BooleanArray::from(data));
            Ok((Field::new(name, DataType::Boolean, true), array))
        }
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            let data: Vec<hdf5::types::VarLenUnicode> = ds.read_1d()?.to_vec();
            let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
            let array: ArrayRef = Arc::new(StringArray::from(
                strings.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ));
            Ok((Field::new(name, DataType::Utf8, true), array))
        }
        TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) => {
            let data: Vec<hdf5::types::VarLenUnicode> = ds.read_1d()?.to_vec();
            let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
            let array: ArrayRef = Arc::new(StringArray::from(
                strings.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ));
            Ok((Field::new(name, DataType::Utf8, true), array))
        }
        other => Err(ConvertError::UnsupportedDtype(format!(
            "column '{name}': unsupported HDF5 type: {other:?}"
        ))),
    }
}

/// Read a categorical column (integer codes + string categories).
fn read_categorical_column(
    group: &hdf5::Group,
    ds: &hdf5::Dataset,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    // Read the integer codes
    let codes: Vec<i32> = read_i32_dataset(ds)?;

    // Try to find categories: first check "categories" attribute on the dataset
    let categories: Vec<String> = if let Ok(cats_attr) = ds.attr("categories") {
        // Categories stored as attribute
        let cats: Vec<hdf5::types::VarLenUnicode> = cats_attr.read_1d()?.to_vec();
        cats.iter().map(|s| s.to_string()).collect()
    } else if let Ok(cats_ds) = group.dataset(&format!("__categories/{name}")) {
        // Old-style: categories in __categories subgroup
        let cats: Vec<hdf5::types::VarLenUnicode> = cats_ds.read_1d()?.to_vec();
        cats.iter().map(|s| s.to_string()).collect()
    } else {
        // Fallback: treat codes as strings
        return Ok((
            Field::new(name, DataType::Int32, true),
            Arc::new(Int32Array::from(codes)),
        ));
    };

    // Build a DictionaryArray
    let keys = Int32Array::from(codes);
    let values = StringArray::from(categories.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values))?;

    let field = Field::new(
        name,
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        true,
    );

    Ok((field, Arc::new(dict)))
}

/// Read obsm embeddings from h5ad file.
pub fn read_obsm(file: &hdf5::File) -> Result<HashMap<String, RecordBatch>, ConvertError> {
    let mut result = HashMap::new();

    let obsm_group = file.group("obsm")?;
    let member_names = obsm_group.member_names()?;

    for name in &member_names {
        match read_obsm_entry(&obsm_group, name) {
            Ok(batch) => {
                result.insert(name.clone(), batch);
            }
            Err(e) => {
                eprintln!("warning: skipping obsm/{name}: {e}");
            }
        }
    }

    Ok(result)
}

fn read_obsm_entry(obsm_group: &hdf5::Group, name: &str) -> Result<RecordBatch, ConvertError> {
    let ds = obsm_group.dataset(name)?;
    let shape = ds.shape();

    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "obsm/{name} is not 2D (shape: {shape:?})"
        )));
    }

    let n_rows = shape[0];
    let n_cols = shape[1];

    // Read as f32 2D
    let flat: Vec<f32> = read_f32_dataset(&ds)?;

    // Build RecordBatch with numbered columns
    let mut fields = Vec::with_capacity(n_cols);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(n_cols);

    for col in 0..n_cols {
        let col_data: Vec<f32> = (0..n_rows).map(|row| flat[row * n_cols + col]).collect();
        fields.push(Field::new(format!("{col}"), DataType::Float32, false));
        arrays.push(Arc::new(Float32Array::from(col_data)));
    }

    let schema = Schema::new(fields);
    Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
}

/// Read uns (unstructured) section from h5ad file as JSON.
pub fn read_uns(file: &hdf5::File) -> Result<serde_json::Value, ConvertError> {
    let uns_group = file.group("uns")?;
    let member_names = uns_group.member_names()?;

    let mut map = serde_json::Map::new();

    for name in &member_names {
        match read_uns_entry(&uns_group, name) {
            Ok(value) => {
                map.insert(name.clone(), value);
            }
            Err(e) => {
                eprintln!("warning: skipping uns/{name}: {e}");
            }
        }
    }

    Ok(serde_json::Value::Object(map))
}

fn read_uns_entry(group: &hdf5::Group, name: &str) -> Result<serde_json::Value, ConvertError> {
    // Try reading as dataset first
    if let Ok(ds) = group.dataset(name) {
        let desc = ds.dtype()?.to_descriptor()?;
        let shape = ds.shape();

        // Scalar
        if shape.is_empty() || (shape.len() == 1 && shape[0] == 1) {
            return match desc {
                TypeDescriptor::Integer(_) => {
                    let v: i64 = ds.read_scalar()?;
                    Ok(serde_json::Value::Number(v.into()))
                }
                TypeDescriptor::Unsigned(_) => {
                    let v: u64 = ds.read_scalar()?;
                    Ok(serde_json::Value::Number(v.into()))
                }
                TypeDescriptor::Float(_) => {
                    let v: f64 = ds.read_scalar()?;
                    Ok(serde_json::json!(v))
                }
                TypeDescriptor::Boolean => {
                    let v: bool = ds.read_scalar()?;
                    Ok(serde_json::Value::Bool(v))
                }
                TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
                    let v: hdf5::types::VarLenUnicode = ds.read_scalar()?;
                    Ok(serde_json::Value::String(v.to_string()))
                }
                TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) => {
                    let v: hdf5::types::VarLenUnicode = ds.read_scalar()?;
                    Ok(serde_json::Value::String(v.to_string()))
                }
                _ => Err(ConvertError::Other(format!(
                    "unsupported uns scalar type: {desc:?}"
                ))),
            };
        }

        // 1D array
        if shape.len() == 1 {
            return match desc {
                TypeDescriptor::Integer(_) => {
                    let data: Vec<i64> = ds.read_1d()?.to_vec();
                    Ok(serde_json::json!(data))
                }
                TypeDescriptor::Float(_) => {
                    let data: Vec<f64> = ds.read_1d()?.to_vec();
                    Ok(serde_json::json!(data))
                }
                TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
                    let data: Vec<hdf5::types::VarLenUnicode> = ds.read_1d()?.to_vec();
                    let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
                    Ok(serde_json::json!(strings))
                }
                _ => Err(ConvertError::Other(format!(
                    "unsupported uns array type: {desc:?}"
                ))),
            };
        }

        return Err(ConvertError::Other(format!(
            "unsupported uns dataset shape: {shape:?}"
        )));
    }

    // Try reading as subgroup → recurse
    if let Ok(subgroup) = group.group(name) {
        let sub_members = subgroup.member_names()?;
        let mut sub_map = serde_json::Map::new();
        for sub_name in &sub_members {
            match read_uns_entry(&subgroup, sub_name) {
                Ok(v) => {
                    sub_map.insert(sub_name.clone(), v);
                }
                Err(e) => {
                    eprintln!("warning: skipping uns/{name}/{sub_name}: {e}");
                }
            }
        }
        return Ok(serde_json::Value::Object(sub_map));
    }

    Err(ConvertError::Other(format!(
        "uns/{name}: not a dataset or group"
    )))
}

/// Read layers from h5ad file.
/// Returns a map of layer_name → (indptr, indices, data, n_obs, n_vars).
pub fn read_layers(file: &hdf5::File) -> Result<HashMap<String, CsrArrays>, ConvertError> {
    let mut result = HashMap::new();
    let layers_group = file.group("layers")?;
    let member_names = layers_group.member_names()?;

    for name in &member_names {
        match read_layer_entry(file, &layers_group, name) {
            Ok(data) => {
                result.insert(name.clone(), data);
            }
            Err(e) => {
                eprintln!("warning: skipping layer '{name}': {e}");
            }
        }
    }

    Ok(result)
}

fn read_layer_entry(
    file: &hdf5::File,
    layers_group: &hdf5::Group,
    name: &str,
) -> Result<CsrArrays, ConvertError> {
    // Each layer is like X: could be sparse group or dense dataset
    let path = format!("layers/{name}");
    if let Ok(group) = layers_group.group(name) {
        // Sparse matrix
        let shape: Vec<i64> = group.attr("shape")?.read_1d()?.to_vec();
        let (n_obs, n_vars) = (shape[0] as usize, shape[1] as usize);

        let is_csc = group
            .attr("encoding-type")
            .ok()
            .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
            .map(|v| v.as_str() == "csc_matrix")
            .unwrap_or(false);

        let indptr = read_i64_dataset(&group.dataset("indptr")?)?;
        let indices = read_i32_dataset(&group.dataset("indices")?)?;
        let data = read_f32_dataset(&group.dataset("data")?)?;

        if is_csc {
            let (csr_indptr, csr_indices, csr_data) =
                csc_to_csr(&indptr, &indices, &data, n_obs, n_vars);
            Ok((csr_indptr, csr_indices, csr_data, n_obs, n_vars))
        } else {
            Ok((indptr, indices, data, n_obs, n_vars))
        }
    } else {
        // Dense dataset
        read_dense_matrix(file, &path)
    }
}
