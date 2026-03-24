// Phase D: Seurat/SCE interop — CSR↔CSC, Arrow→data.frame
//
// This module provides the minimum viable interop functions needed by Phase B
// (reader.rs). Full Seurat/SCE conversion will be added in Phase D.

use arrow::array::{
    Array, AsArray, RecordBatch,
};
use arrow::datatypes::DataType;
use extendr_api::prelude::*;
use scx_sparse::ScxCsr;

// ─── Arrow RecordBatch → R data.frame ────────────────────────────────────────

/// Convert an Arrow RecordBatch to an R data.frame.
///
/// Column type mapping:
///   Arrow Utf8/LargeUtf8      → R character vector
///   Arrow Dictionary(Utf8)    → R factor (levels from dictionary, indices from values)
///   Arrow Int8/16/32          → R integer vector
///   Arrow Int64               → R double vector (R has no native i64)
///   Arrow UInt8/16/32         → R integer vector (safe range)
///   Arrow UInt64              → R double vector
///   Arrow Float32/64          → R double vector
///   Arrow Boolean             → R logical vector
pub fn record_batch_to_dataframe(batch: &RecordBatch) -> Result<Robj> {
    let schema = batch.schema();
    let n_cols = schema.fields().len();

    // Build a named list of R vectors, one per column
    let mut columns: Vec<(&str, Robj)> = Vec::with_capacity(n_cols);

    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        let name = field.name().as_str();
        let robj = arrow_column_to_robj(col, field.data_type())?;
        columns.push((name, robj));
    }

    // Convert named list to data.frame
    let list = List::from_pairs(columns);
    // Set class to "data.frame" and row.names
    let n_rows = batch.num_rows() as i32;
    let robj: Robj = list.into();

    // Use R to construct a proper data.frame with row.names
    R!("{ x <- {{robj}}; class(x) <- 'data.frame'; attr(x, 'row.names') <- seq_len({{n_rows}}); x }")
        .map_err(|e| Error::Other(format!("data.frame construction failed: {}", e)))
}

/// Convert a single Arrow array column to an R vector.
fn arrow_column_to_robj(col: &dyn Array, dtype: &DataType) -> Result<Robj> {
    match dtype {
        // String types → character vector
        DataType::Utf8 => {
            let arr = col.as_string::<i32>();
            let strings: Vec<Option<String>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i).to_string())
                    }
                })
                .collect();
            // Convert to R character vector, mapping None → NA
            let rvec: Vec<Option<&str>> = strings
                .iter()
                .map(|s| s.as_deref())
                .collect();
            Ok(rvec.into_robj())
        }
        DataType::LargeUtf8 => {
            let arr = col.as_string::<i64>();
            let strings: Vec<Option<String>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i).to_string())
                    }
                })
                .collect();
            let rvec: Vec<Option<&str>> = strings
                .iter()
                .map(|s| s.as_deref())
                .collect();
            Ok(rvec.into_robj())
        }

        // Dictionary → R factor
        DataType::Dictionary(key_type, value_type) => {
            dictionary_to_factor(col, key_type, value_type)
        }

        // Integer types → R integer (i32)
        DataType::Int8 => {
            let arr = col.as_primitive::<arrow::datatypes::Int8Type>();
            let vals: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as i32) }
                })
                .collect();
            Ok(vals.into_robj())
        }
        DataType::Int16 => {
            let arr = col.as_primitive::<arrow::datatypes::Int16Type>();
            let vals: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as i32) }
                })
                .collect();
            Ok(vals.into_robj())
        }
        DataType::Int32 => {
            let arr = col.as_primitive::<arrow::datatypes::Int32Type>();
            let vals: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i)) }
                })
                .collect();
            Ok(vals.into_robj())
        }
        // Int64 → R double (R has no native i64)
        DataType::Int64 => {
            let arr = col.as_primitive::<arrow::datatypes::Int64Type>();
            let vals: Vec<Option<f64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as f64) }
                })
                .collect();
            Ok(vals.into_robj())
        }

        // Unsigned integer types → R integer (safe range for u8/u16; u32 may overflow)
        DataType::UInt8 => {
            let arr = col.as_primitive::<arrow::datatypes::UInt8Type>();
            let vals: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as i32) }
                })
                .collect();
            Ok(vals.into_robj())
        }
        DataType::UInt16 => {
            let arr = col.as_primitive::<arrow::datatypes::UInt16Type>();
            let vals: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as i32) }
                })
                .collect();
            Ok(vals.into_robj())
        }
        DataType::UInt32 => {
            // u32 can exceed i32::MAX; use f64 to be safe
            let arr = col.as_primitive::<arrow::datatypes::UInt32Type>();
            let vals: Vec<Option<f64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as f64) }
                })
                .collect();
            Ok(vals.into_robj())
        }
        DataType::UInt64 => {
            let arr = col.as_primitive::<arrow::datatypes::UInt64Type>();
            let vals: Vec<Option<f64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as f64) }
                })
                .collect();
            Ok(vals.into_robj())
        }

        // Float types → R double
        DataType::Float32 => {
            let arr = col.as_primitive::<arrow::datatypes::Float32Type>();
            let vals: Vec<Option<f64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i) as f64) }
                })
                .collect();
            Ok(vals.into_robj())
        }
        DataType::Float64 => {
            let arr = col.as_primitive::<arrow::datatypes::Float64Type>();
            let vals: Vec<Option<f64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i)) }
                })
                .collect();
            Ok(vals.into_robj())
        }

        // Boolean → R logical
        DataType::Boolean => {
            let arr = col.as_boolean();
            let vals: Vec<Option<bool>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i)) }
                })
                .collect();
            Ok(vals.into_robj())
        }

        // Null → NA vector
        DataType::Null => {
            let n = col.len();
            let vals: Vec<Option<bool>> = vec![None; n];
            Ok(vals.into_robj())
        }

        other => Err(Error::Other(format!(
            "unsupported Arrow type for R conversion: {:?}",
            other
        ))),
    }
}

/// Convert a Dictionary-encoded Arrow column to an R factor.
///
/// Dictionary encoding in Arrow maps to R's factor type:
///   - dictionary values → factor levels
///   - dictionary indices → factor integer codes (1-based in R)
fn dictionary_to_factor(
    col: &dyn Array,
    key_type: &DataType,
    value_type: &DataType,
) -> Result<Robj> {
    // We support Dictionary<Int8/16/32, Utf8> which is the common h5ad categorical pattern
    match (key_type, value_type) {
        (DataType::Int8, DataType::Utf8) => {
            typed_dict_to_factor::<arrow::datatypes::Int8Type>(col)
        }
        (DataType::Int16, DataType::Utf8) => {
            typed_dict_to_factor::<arrow::datatypes::Int16Type>(col)
        }
        (DataType::Int32, DataType::Utf8) => {
            typed_dict_to_factor::<arrow::datatypes::Int32Type>(col)
        }
        _ => Err(Error::Other(format!(
            "unsupported Dictionary key/value types: {:?}/{:?}",
            key_type, value_type
        ))),
    }
}

/// Helper: extract factor levels and codes from a typed DictionaryArray.
fn typed_dict_to_factor<K>(col: &dyn Array) -> Result<Robj>
where
    K: arrow::datatypes::ArrowDictionaryKeyType,
    K::Native: TryInto<i32>,
{
    use arrow::array::AsArray;

    let dict_arr = col.as_any_dictionary();

    // Extract levels from the dictionary values (Utf8)
    let values = dict_arr
        .values()
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or_else(|| Error::Other("Dictionary values are not Utf8".into()))?;
    let levels: Vec<String> = (0..values.len())
        .map(|i| values.value(i).to_string())
        .collect();

    // Extract codes (1-based for R factors, NA for nulls)
    let keys = dict_arr.keys();
    let codes: Vec<Option<i32>> = (0..keys.len())
        .map(|i| {
            if dict_arr.is_null(i) {
                None
            } else {
                // Arrow indices are 0-based, R factor codes are 1-based
                let key_val = keys.as_primitive::<K>().value(i);
                match key_val.try_into() {
                    Ok(v) => Some(v + 1), // 0-based → 1-based
                    Err(_) => None,
                }
            }
        })
        .collect();

    // Build factor in R: integer vector with "levels" and "class" attributes
    let levels_robj: Robj = levels.into_robj();
    let codes_robj: Robj = codes.into_robj();
    R!("{ x <- {{codes_robj}}; attr(x, 'levels') <- {{levels_robj}}; class(x) <- 'factor'; x }")
        .map_err(|e| Error::Other(format!("factor construction failed: {}", e)))
}

// ─── ScxCsr (CSR) → dgCMatrix (CSC) ─────────────────────────────────────────

/// Convert ScxCsr (CSR, i64/i32/f32) → R dgCMatrix (CSC, i32/i32/f64).
///
/// Steps:
///   1. Transpose CSR → CSC arrays (indptr_csc, indices_csc, data_csc)
///      - Two-pass algorithm: count per-column nnz, then scatter
///   2. Cast types: indptr i64→i32 (error if >2B nnz), data f32→f64 (lossless)
///   3. Construct dgCMatrix via R: new("dgCMatrix", i=indices, p=indptr, x=data, Dim=c(m,n))
///
/// Note on 0-indexing: dgCMatrix uses 0-based indices (same as CSC),
/// so no index adjustment is needed after transpose.
pub fn csr_to_dgcmatrix(csr: &ScxCsr) -> Result<Robj> {
    let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
    let nnz = csr.nnz();

    // Validate i32 range
    if nnz > i32::MAX as usize {
        return Err(Error::Other(format!(
            "nnz {} exceeds i32::MAX, cannot create dgCMatrix",
            nnz
        )));
    }

    // Handle empty matrix
    if nnz == 0 {
        let indptr_csc = vec![0i32; n_cols + 1];
        let indices_csc: Vec<i32> = vec![];
        let data_csc: Vec<f64> = vec![];
        let dim_m = n_rows as i32;
        let dim_n = n_cols as i32;
        return R!("methods::new('dgCMatrix',
             i = {{indices_csc}},
             p = {{indptr_csc}},
             x = {{data_csc}},
             Dim = c({{dim_m}}, {{dim_n}})
        )")
        .map_err(|e| Error::Other(format!("dgCMatrix construction failed: {}", e)));
    }

    // 1. CSR → CSC transpose (two-pass scatter)
    let mut col_counts = vec![0i32; n_cols];
    for &idx in &csr.indices {
        col_counts[idx as usize] += 1;
    }

    let mut indptr_csc = vec![0i32; n_cols + 1];
    for j in 0..n_cols {
        indptr_csc[j + 1] = indptr_csc[j] + col_counts[j];
    }

    let mut indices_csc = vec![0i32; nnz];
    let mut data_csc = vec![0.0f64; nnz];
    let mut write_pos = indptr_csc[..n_cols].to_vec();

    for i in 0..n_rows {
        let row_start = csr.indptr[i] as usize;
        let row_end = csr.indptr[i + 1] as usize;
        for k in row_start..row_end {
            let col = csr.indices[k] as usize;
            let pos = write_pos[col] as usize;
            indices_csc[pos] = i as i32;
            data_csc[pos] = csr.data[k] as f64; // f32 → f64 widening
            write_pos[col] += 1;
        }
    }

    // 2. Construct dgCMatrix in R via methods::new()
    let dim_m = n_rows as i32;
    let dim_n = n_cols as i32;
    R!("methods::new('dgCMatrix',
         i = {{indices_csc}},
         p = {{indptr_csc}},
         x = {{data_csc}},
         Dim = c({{dim_m}}, {{dim_n}})
    )")
    .map_err(|e| Error::Other(format!("dgCMatrix construction failed: {}", e)))
}

// ─── Phase D stubs (Seurat/SCE interop) ──────────────────────────────────────
// These will be implemented in Phase D once Phase B is verified.

// pub fn to_seurat_v5(result: &QueryResult) -> Result<Robj> { ... }
// pub fn to_sce(result: &QueryResult) -> Result<Robj> { ... }
// pub fn from_seurat(seurat_obj: Robj, output_path: &str) -> Result<()> { ... }
// pub fn from_sce(sce_obj: Robj, output_path: &str) -> Result<()> { ... }
