// Phase D: Seurat/SCE interop — CSR↔CSC, Arrow→data.frame
//
// This module provides the minimum viable interop functions needed by Phase B
// (reader.rs). Full Seurat/SCE conversion will be added in Phase D.

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::DataType;
use extendr_api::prelude::*;
use scx_codec::ValueEncoding;
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

/// Helper: convert an Arrow primitive column to an R vector with null handling.
fn primitive_to_robj<T, R>(col: &dyn Array, convert: impl Fn(T::Native) -> R) -> Result<Robj>
where
    T: arrow::datatypes::ArrowPrimitiveType,
    Vec<Option<R>>: IntoRobj,
{
    let arr = col.as_primitive::<T>();
    let vals: Vec<Option<R>> = (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(convert(arr.value(i)))
            }
        })
        .collect();
    Ok(vals.into_robj())
}

/// Helper: convert an Arrow string column to an R character vector with null handling.
fn string_to_robj<O: arrow::array::OffsetSizeTrait>(col: &dyn Array) -> Result<Robj> {
    let arr = col.as_string::<O>();
    let strings: Vec<Option<String>> = (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i).to_string())
            }
        })
        .collect();
    let rvec: Vec<Option<&str>> = strings.iter().map(|s| s.as_deref()).collect();
    Ok(rvec.into_robj())
}

/// Convert a single Arrow array column to an R vector.
fn arrow_column_to_robj(col: &dyn Array, dtype: &DataType) -> Result<Robj> {
    use arrow::datatypes::*;
    match dtype {
        // String types → character vector
        DataType::Utf8 => string_to_robj::<i32>(col),
        DataType::LargeUtf8 => string_to_robj::<i64>(col),

        // Dictionary → R factor
        DataType::Dictionary(key_type, value_type) => {
            dictionary_to_factor(col, key_type, value_type)
        }

        // Integer types → R integer (i32)
        DataType::Int8 => primitive_to_robj::<Int8Type, i32>(col, |v| v as i32),
        DataType::Int16 => primitive_to_robj::<Int16Type, i32>(col, |v| v as i32),
        DataType::Int32 => primitive_to_robj::<Int32Type, i32>(col, |v| v),
        // Int64 → R double (R has no native i64)
        DataType::Int64 => primitive_to_robj::<Int64Type, f64>(col, |v| v as f64),

        // Unsigned integer types → R integer (safe range for u8/u16)
        DataType::UInt8 => primitive_to_robj::<UInt8Type, i32>(col, |v| v as i32),
        DataType::UInt16 => primitive_to_robj::<UInt16Type, i32>(col, |v| v as i32),
        // u32 can exceed i32::MAX; use f64 to be safe
        DataType::UInt32 => primitive_to_robj::<UInt32Type, f64>(col, |v| v as f64),
        DataType::UInt64 => primitive_to_robj::<UInt64Type, f64>(col, |v| v as f64),

        // Float types → R double
        DataType::Float32 => primitive_to_robj::<Float32Type, f64>(col, |v| v as f64),
        DataType::Float64 => primitive_to_robj::<Float64Type, f64>(col, |v| v),

        // Boolean → R logical
        DataType::Boolean => {
            let arr = col.as_boolean();
            let vals: Vec<Option<bool>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect();
            Ok(vals.into_robj())
        }

        // Null → NA vector
        DataType::Null => {
            let vals: Vec<Option<bool>> = vec![None; col.len()];
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
        (DataType::Int8, DataType::Utf8) => typed_dict_to_factor::<arrow::datatypes::Int8Type>(col),
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

// ─── Phase D: Seurat v5 & SingleCellExperiment Interop ───────────────────────

use scx_engine::pipeline::QueryResult;

/// Create a Seurat v5 object from SCX query results.
///
/// ScxCsr → dgCMatrix (CSR→CSC, cells × genes) → t(dgCMatrix) (genes × cells).
/// obs → Seurat meta.data (R data.frame).
/// var → Seurat feature metadata.
///
/// Orientation chain:
///   SCX CSR: cells × genes (n_obs rows, n_vars cols)
///   csr_to_dgcmatrix(): CSC, still cells × genes (n_obs rows, n_vars cols)
///   Matrix::t(): CSC, genes × cells (n_vars rows, n_obs cols) ← Seurat expects this
///
/// Requires: Seurat >= 5.0.0 (listed in Suggests)
pub fn to_seurat_v5(result: &QueryResult) -> Result<Robj> {
    let dgc = csr_to_dgcmatrix(&result.x)?;
    let obs_df = record_batch_to_dataframe(&result.obs)?;
    let var_df = record_batch_to_dataframe(&result.var)?;

    R!("
        if (!requireNamespace('Seurat', quietly = TRUE))
            stop('Seurat >= 5.0.0 is required for to_seurat()')
        counts_t <- Matrix::t({{dgc}})
        seu <- Seurat::CreateSeuratObject(counts = counts_t)
        seu@meta.data <- {{obs_df}}
        seu[['RNA']]@meta.data <- {{var_df}}
        seu
    ")
    .map_err(|e| Error::Other(format!("Seurat construction failed: {}", e)))
}

/// Create a SingleCellExperiment from SCX query results.
///
/// Same dgCMatrix construction + transpose as to_seurat_v5.
/// SCE expects assay matrices as genes × cells (rows × cols).
/// obs → colData (columns = cells)
/// var → rowData (rows = genes)
///
/// Requires: SingleCellExperiment (listed in Suggests)
pub fn to_sce(result: &QueryResult) -> Result<Robj> {
    let dgc = csr_to_dgcmatrix(&result.x)?;
    let obs_df = record_batch_to_dataframe(&result.obs)?;
    let var_df = record_batch_to_dataframe(&result.var)?;

    R!("
        if (!requireNamespace('SingleCellExperiment', quietly = TRUE))
            stop('SingleCellExperiment is required for to_sce()')
        counts_t <- Matrix::t({{dgc}})
        SingleCellExperiment::SingleCellExperiment(
            assays = list(counts = counts_t),
            colData = S4Vectors::DataFrame({{obs_df}}),
            rowData = S4Vectors::DataFrame({{var_df}})
        )
    ")
    .map_err(|e| Error::Other(format!("SCE construction failed: {}", e)))
}

// ─── CSC (dgCMatrix) → CSR transpose ────────────────────────────────────────

/// Extract dgCMatrix slots and transpose CSC → CSR.
///
/// dgCMatrix is an S4 object with slots:
///   @i  — integer (0-based row indices)
///   @p  — integer (column pointers, length n_cols + 1)
///   @x  — double (values)
///   @Dim — integer[2] (n_rows, n_cols)
///
/// Returns (indptr_csr as Vec<u64>, indices_csr as Vec<u32>, values_bytes as Vec<u8>)
/// after transposing CSC → CSR, where:
///   - CSC "rows" (dimension 0) become CSR rows
///   - CSC "columns" (dimension 1) become CSR columns
///
/// Since Seurat/SCE store genes × cells, we transpose to get cells × genes (SCX layout).
///
/// Returns (indptr, indices, value_bytes, n_rows, n_cols, encoding).
type CsrData = (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize, ValueEncoding);

fn dgcmatrix_to_csr(dgc: &Robj) -> Result<CsrData> {
    // Extract dgCMatrix slots via R
    let dim_robj =
        R!("{{dgc}}@Dim").map_err(|e| Error::Other(format!("failed to get Dim: {}", e)))?;
    let dim: Vec<i32> = dim_robj
        .as_integer_slice()
        .ok_or_else(|| Error::Other("Dim is not integer".into()))?
        .to_vec();
    if dim.len() != 2 {
        return Err(Error::Other(format!(
            "Dim has {} elements, expected 2",
            dim.len()
        )));
    }
    let n_rows = dim[0] as usize; // genes in Seurat/SCE (becomes cells after transpose)
    let n_cols = dim[1] as usize; // cells in Seurat/SCE (becomes genes after transpose)

    let indices_robj =
        R!("{{dgc}}@i").map_err(|e| Error::Other(format!("failed to get @i: {}", e)))?;
    let csc_indices: Vec<i32> = indices_robj
        .as_integer_slice()
        .ok_or_else(|| Error::Other("@i is not integer".into()))?
        .to_vec();

    let indptr_robj =
        R!("{{dgc}}@p").map_err(|e| Error::Other(format!("failed to get @p: {}", e)))?;
    let csc_indptr: Vec<i32> = indptr_robj
        .as_integer_slice()
        .ok_or_else(|| Error::Other("@p is not integer".into()))?
        .to_vec();

    let values_robj =
        R!("{{dgc}}@x").map_err(|e| Error::Other(format!("failed to get @x: {}", e)))?;
    let csc_values: Vec<f64> = values_robj
        .as_real_slice()
        .ok_or_else(|| Error::Other("@x is not double".into()))?
        .to_vec();

    let nnz = csc_values.len();

    // Transpose CSC (genes × cells) → CSR (cells × genes)
    // In the transposed layout: rows = cells (n_cols), cols = genes (n_rows)
    let csr_n_rows = n_cols; // cells
    let csr_n_cols = n_rows; // genes

    // Count nnz per row in CSR (= per column index in CSC, which is the row index @i)
    let mut row_counts = vec![0u64; csr_n_rows];
    // In CSC→CSR transpose: CSC columns become CSR columns (genes), CSC row indices become CSR row indices (cells)
    // Actually no: CSC(genes×cells) has columns=cells, rows=genes.
    // Transposing gives us CSR(cells×genes) where rows=cells, cols=genes.
    // In CSC format: column j (cell j) has entries at rows csc_indices[csc_indptr[j]..csc_indptr[j+1]].
    // In CSR format: row j (cell j) has entries at columns = the gene indices.
    // So CSC column j becomes CSR row j. For each entry in CSC column j at row i,
    // we get CSR row j, column i.

    // Count entries per CSR row (= per CSC column)
    for j in 0..n_cols {
        let start = csc_indptr[j] as usize;
        let end = csc_indptr[j + 1] as usize;
        row_counts[j] = (end - start) as u64;
    }

    // Build CSR indptr
    let mut csr_indptr = vec![0u64; csr_n_rows + 1];
    for i in 0..csr_n_rows {
        csr_indptr[i + 1] = csr_indptr[i] + row_counts[i];
    }

    // Scatter values into CSR arrays
    let mut csr_indices = vec![0u32; nnz];
    let mut csr_values_f64 = vec![0.0f64; nnz];
    let mut write_pos: Vec<u64> = csr_indptr[..csr_n_rows].to_vec();

    for j in 0..n_cols {
        // CSC column j → CSR row j
        let start = csc_indptr[j] as usize;
        let end = csc_indptr[j + 1] as usize;
        for k in start..end {
            let gene_idx = csc_indices[k] as u32; // CSC row index = gene
            let pos = write_pos[j] as usize;
            csr_indices[pos] = gene_idx;
            csr_values_f64[pos] = csc_values[k];
            write_pos[j] += 1;
        }
    }

    // Detect value encoding: integer-like values get uint8/uint16/uint32,
    // otherwise fall back to float32. Mirrors scx-cli detect_value_encoding_only().
    let all_integer = csr_values_f64
        .iter()
        .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor());

    let (values_bytes, value_encoding) = if all_integer {
        let max_val: f64 = csr_values_f64.iter().copied().fold(0.0f64, f64::max);
        if max_val <= 255.0 {
            (
                csr_values_f64.iter().map(|&v| v as u8).collect(),
                ValueEncoding::Uint8,
            )
        } else if max_val <= 65535.0 {
            let mut buf = Vec::with_capacity(csr_values_f64.len() * 2);
            for &v in &csr_values_f64 {
                buf.extend_from_slice(&(v as u16).to_le_bytes());
            }
            (buf, ValueEncoding::Uint16)
        } else {
            let mut buf = Vec::with_capacity(csr_values_f64.len() * 4);
            for &v in &csr_values_f64 {
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
            (buf, ValueEncoding::Uint32)
        }
    } else {
        let mut buf = Vec::with_capacity(csr_values_f64.len() * 4);
        for &v in &csr_values_f64 {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        (buf, ValueEncoding::Float32)
    };

    Ok((
        csr_indptr,
        csr_indices,
        values_bytes,
        csr_n_rows,
        csr_n_cols,
        value_encoding,
    ))
}

/// Convert an R data.frame to an Arrow RecordBatch for writing obs/var.
///
/// This is the reverse of `record_batch_to_dataframe`.
/// Extracts column names and values, creating Utf8 columns for character vectors
/// and Float64 columns for numeric vectors.
fn dataframe_to_record_batch(df: &Robj) -> Result<arrow::array::RecordBatch> {
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    let names_robj =
        R!("names({{df}})").map_err(|e| Error::Other(format!("failed to get names: {}", e)))?;
    let col_names: Vec<String> = names_robj
        .as_str_iter()
        .ok_or_else(|| Error::Other("names not character".into()))?
        .map(|s| s.to_string())
        .collect();

    let n_cols = col_names.len();
    let mut fields = Vec::with_capacity(n_cols);
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(n_cols);

    for (i, name) in col_names.iter().enumerate() {
        let col_idx = (i + 1) as i32; // R is 1-based
        let col_robj = R!("{{df}}[[{{col_idx}}]]")
            .map_err(|e| Error::Other(format!("failed to get column {}: {}", name, e)))?;

        let is_char = R!("is.character({{df}}[[{{col_idx}}]])")
            .map_err(|e| Error::Other(format!("is.character check failed: {}", e)))?;
        let is_character = is_char
            .as_logical_slice()
            .and_then(|s| s.first().map(|&b| b.is_true()))
            .unwrap_or(false);

        let is_fac = R!("is.factor({{df}}[[{{col_idx}}]])")
            .map_err(|e| Error::Other(format!("is.factor check failed: {}", e)))?;
        let is_factor = is_fac
            .as_logical_slice()
            .and_then(|s| s.first().map(|&b| b.is_true()))
            .unwrap_or(false);

        if is_character || is_factor {
            // Convert to character vector (handles factors too)
            let char_robj = R!("as.character({{df}}[[{{col_idx}}]])")
                .map_err(|e| Error::Other(format!("as.character failed: {}", e)))?;
            let strings: Vec<Option<String>> = char_robj
                .as_str_iter()
                .ok_or_else(|| Error::Other(format!("column {} not iterable as str", name)))?
                .map(|s| Some(s.to_string()))
                .collect();
            let arr = StringArray::from(strings.iter().map(|s| s.as_deref()).collect::<Vec<_>>());
            fields.push(Field::new(name, DataType::Utf8, true));
            arrays.push(Arc::new(arr));
        } else {
            // Numeric → f64 array
            let vals: Vec<Option<f64>> = col_robj
                .as_real_slice()
                .ok_or_else(|| Error::Other(format!("column {} not numeric", name)))?
                .iter()
                .map(|&v| if v.is_nan() { None } else { Some(v) })
                .collect();
            let arr = Float64Array::from(vals);
            fields.push(Field::new(name, DataType::Float64, true));
            arrays.push(Arc::new(arr));
        }
    }

    let schema = Arc::new(Schema::new(fields));
    arrow::array::RecordBatch::try_new(schema, arrays)
        .map_err(|e| Error::Other(format!("RecordBatch construction failed: {}", e)))
}

// ─── Shared SCX writer helper ────────────────────────────────────────────────

/// Parse a codec name from R to an Option<CodecId>.
fn parse_codec_r(codec: Option<&str>) -> Result<Option<scx_codec::CodecId>> {
    use scx_codec::CodecId;
    match codec {
        None | Some("auto") => Ok(None),
        Some("none") => Ok(Some(CodecId::None)),
        Some("scx1") => Ok(Some(CodecId::Scx1)),
        Some("zstd") => Ok(Some(CodecId::Zstd)),
        Some("lz4") => Ok(Some(CodecId::Lz4Shuffle)),
        Some("pcodec") => Ok(Some(CodecId::Pcodec)),
        Some(other) => Err(Error::Other(format!(
            "Unknown codec: '{}'. Use 'auto', 'none', 'scx1', 'zstd', 'lz4', or 'pcodec'.",
            other
        ))),
    }
}

/// Write CSR data to an SCX file with multi-shard splitting and auto-codec.
///
/// Shared by `from_seurat` and `from_sce` — both extract R objects into
/// the same (indptr, indices, values_bytes) representation, then delegate here.
///
/// `csc_always`: when true, also emits a CSC sidecar (multi-shard
/// column-major). Uses the streaming transpose iterator with the same
/// `csc_cols_per_shard` cap as `scx convert` and `pyscx.from_anndata`.
/// Note: the "write dgCMatrix arrays directly via `write_csc_shard`"
/// short-cut suggested in the original spec is not applicable —
/// dgCMatrix is genes×cells while SCX-CSC is cells×genes (column-major
/// over genes), so a full transpose is still required.
#[allow(clippy::too_many_arguments)]
fn write_csr_to_scx(
    output_path: &str,
    csr_indptr: &[u64],
    csr_indices: &[u32],
    values_bytes: &[u8],
    value_encoding: ValueEncoding,
    n_obs: usize,
    n_vars: usize,
    obs_batch: &RecordBatch,
    var_batch: &RecordBatch,
    explicit_codec: Option<scx_codec::CodecId>,
    csc_always: bool,
    csc_cols_per_shard: usize,
) -> Result<()> {
    use scx_codec::CodecId;
    use scx_format::header::FileHeader;
    use scx_format::select_codec;
    use scx_format::writer::ScxWriter;

    let nnz = *csr_indptr.last().unwrap_or(&0);
    let shard_target_rows: usize = 16384;
    let n_shards = n_obs.div_ceil(shard_target_rows.max(1));

    let header = FileHeader {
        magic: scx_format::MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: n_vars as u64,
        nnz,
        n_csr_shards: n_shards as u32,
        n_csc_shards: 0,
        shard_target_rows: shard_target_rows as u32,
        codec_id: CodecId::None as u8,
        index_dtype: if n_vars <= 65535 { 0 } else { 1 },
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        reserved: [0u8; 132],
    };

    let mut writer = ScxWriter::new(output_path, header)
        .map_err(|e| Error::Other(format!("ScxWriter::new failed: {}", e)))?;

    writer
        .write_obs(obs_batch)
        .map_err(|e| Error::Other(format!("write_obs failed: {}", e)))?;
    writer
        .write_var(var_batch)
        .map_err(|e| Error::Other(format!("write_var failed: {}", e)))?;

    let bw = value_encoding.byte_width();
    let mut row_start: usize = 0;
    // Track the last shard's resolved codec; reused for the optional
    // CSC sidecar so the two layouts share encoder semantics.
    let mut last_csr_codec = CodecId::None;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        let base = csr_indptr[row_start];
        let shard_indptr: Vec<u64> = csr_indptr[row_start..=row_end]
            .iter()
            .map(|&v| v - base)
            .collect();

        let nnz_start = base as usize;
        let nnz_end = csr_indptr[row_end] as usize;
        let shard_indices: Vec<u32> = csr_indices[nnz_start..nnz_end].to_vec();
        let shard_values = &values_bytes[nnz_start * bw..nnz_end * bw];

        let codec = match explicit_codec {
            Some(c) => {
                if c == CodecId::Scx1 && !value_encoding.is_integer() {
                    CodecId::Zstd
                } else {
                    c
                }
            }
            None => select_codec(shard_values, value_encoding),
        };
        last_csr_codec = codec;

        writer
            .write_csr_shard(
                &shard_indptr,
                &shard_indices,
                shard_values,
                codec,
                value_encoding,
                row_start as u64,
            )
            .map_err(|e| Error::Other(format!("write_csr_shard failed: {}", e)))?;

        row_start = row_end;
    }

    // Optional CSC sidecar — streaming transpose over the full
    // in-memory CSR matrix.
    if csc_always && n_obs > 0 && n_vars > 0 {
        write_csc_shards_from_csr_r(
            &mut writer,
            csr_indptr,
            csr_indices,
            values_bytes,
            value_encoding,
            n_obs,
            n_vars,
            last_csr_codec,
            csc_cols_per_shard,
        )?;
    }

    writer
        .finish()
        .map_err(|e| Error::Other(format!("finish failed: {}", e)))?;

    Ok(())
}

/// Memory budget for the convert-time streaming CSR→CSC transpose
/// (4 GiB, matches scx-cli and pyscx).
const RSCX_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Streaming CSR → CSC transpose over the in-memory `(csr_indptr,
/// csr_indices, values_bytes)` arrays, writing each chunk as one CSC
/// shard. Mirrors the helpers in `scx-cli::convert` and
/// `pyscx::anndata` so all three import paths produce structurally
/// identical CSC sidecars.
#[allow(clippy::too_many_arguments)]
fn write_csc_shards_from_csr_r(
    writer: &mut scx_format::writer::ScxWriter,
    csr_indptr: &[u64],
    csr_indices: &[u32],
    values_bytes: &[u8],
    value_encoding: ValueEncoding,
    n_obs: usize,
    n_vars: usize,
    codec_id: scx_codec::CodecId,
    csc_cols_per_shard: usize,
) -> Result<()> {
    // Decode raw value bytes to f32 once (the streaming iterator works
    // on f32 data internally).
    let nnz = *csr_indptr.last().unwrap_or(&0) as usize;
    let bw = value_encoding.byte_width();
    let data_f32 = decode_values_to_f32(&values_bytes[..nnz * bw], value_encoding)?;
    let indptr_i64: Vec<i64> = csr_indptr.iter().map(|&v| v as i64).collect();
    let indices_i32: Vec<i32> = csr_indices.iter().map(|&v| v as i32).collect();

    let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr_i64, indices_i32, data_f32);
    let shards = std::slice::from_ref(&csr);

    let mut iter = scx_sparse::streaming_csr_to_csc_iter_with_cap(
        shards,
        n_obs,
        n_vars,
        RSCX_CSC_MEMORY_BYTES,
        csc_cols_per_shard,
    )
    .map_err(|e| Error::Other(format!("CSC transpose: {}", e)))?;

    loop {
        let col_start = iter.current_col_start() as u64;
        let chunk = match iter.next() {
            Some(c) => c.map_err(|e| Error::Other(format!("CSC chunk: {}", e)))?,
            None => break,
        };
        let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
        let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
        let raw_values = encode_values_from_f32(&chunk.data, value_encoding)?;
        writer
            .write_csc_shard(
                &csc_indptr_u64,
                &csc_indices_u32,
                &raw_values,
                codec_id,
                value_encoding,
                col_start,
            )
            .map_err(|e| Error::Other(format!("write_csc_shard failed: {}", e)))?;
    }
    Ok(())
}

/// Decode raw little-endian value bytes back to f32.
fn decode_values_to_f32(values_bytes: &[u8], encoding: ValueEncoding) -> Result<Vec<f32>> {
    match encoding {
        ValueEncoding::Uint8 => Ok(values_bytes.iter().map(|&b| b as f32).collect()),
        ValueEncoding::Uint16 => {
            let n = values_bytes.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let v = u16::from_le_bytes([values_bytes[2 * i], values_bytes[2 * i + 1]]);
                out.push(v as f32);
            }
            Ok(out)
        }
        ValueEncoding::Uint32 => {
            let n = values_bytes.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let v = u32::from_le_bytes([
                    values_bytes[4 * i],
                    values_bytes[4 * i + 1],
                    values_bytes[4 * i + 2],
                    values_bytes[4 * i + 3],
                ]);
                out.push(v as f32);
            }
            Ok(out)
        }
        ValueEncoding::Float32 => {
            let n = values_bytes.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let v = f32::from_le_bytes([
                    values_bytes[4 * i],
                    values_bytes[4 * i + 1],
                    values_bytes[4 * i + 2],
                    values_bytes[4 * i + 3],
                ]);
                out.push(v);
            }
            Ok(out)
        }
        ValueEncoding::Float16 => Err(Error::Other(
            "Float16 value encoding not supported by rscx CSC sidecar".into(),
        )),
    }
}

/// Encode f32 values back to the requested LE byte representation.
fn encode_values_from_f32(values: &[f32], encoding: ValueEncoding) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(values.len() * encoding.byte_width());
    for &v in values {
        encoding
            .encode_f32(&mut buf, v)
            .map_err(|e| Error::Other(format!("encode_f32 failed: {}", e)))?;
    }
    Ok(buf)
}

// ─── Import from Seurat/SCE ─────────────────────────────────────────────────

/// Import a Seurat object to an SCX file.
///
/// Extracts the counts dgCMatrix (CSC, genes × cells), transposes to CSR
/// (cells × genes), and writes via ScxWriter.
/// Extracts meta.data → obs, feature metadata → var.
///
/// `csc`: when `TRUE`, also writes a CSC (column-major) sidecar.
/// Default `FALSE` matches `scx convert --csc off`.
///
/// `csc_cols_per_shard`: columns per emitted CSC shard (default 5000).
/// Pass `0L` to disable the cap. Ignored when `csc = FALSE`.
/// @export
#[extendr]
pub fn from_seurat(
    seurat_obj: Robj,
    output_path: &str,
    codec: Option<&str>,
    csc: Option<bool>,
    csc_cols_per_shard: Option<i32>,
) -> Result<()> {
    let explicit_codec = parse_codec_r(codec)?;
    let csc_always = csc.unwrap_or(false);
    let csc_cols_per_shard = csc_cols_per_shard
        .map(|v| {
            if v < 0 {
                Err(Error::Other(format!(
                    "csc_cols_per_shard must be >= 0, got {}",
                    v
                )))
            } else {
                Ok(v as usize)
            }
        })
        .transpose()?
        .unwrap_or(5000);
    // Single R!() call — moves seurat_obj once, returns a lightweight list
    let parts = R!("
        if (!requireNamespace('Seurat', quietly = TRUE))
            stop('Seurat >= 5.0.0 is required for from_seurat()')
        seu <- {{seurat_obj}}
        counts <- Seurat::GetAssayData(seu, layer = 'counts')
        obs_df <- seu@meta.data
        var_df <- tryCatch(seu[['RNA']]@meta.data,
            error = function(e) data.frame(gene_id = rownames(seu)))
        list(counts = counts, obs = obs_df, var = var_df)
    ")
    .map_err(|e| Error::Other(format!("failed to extract Seurat data: {}", e)))?;

    // Unpack via Robj::dollar() — no additional R!() calls or clones
    let counts = parts
        .dollar("counts")
        .map_err(|e| Error::Other(format!("failed to get counts: {}", e)))?;
    let obs_df = parts
        .dollar("obs")
        .map_err(|e| Error::Other(format!("failed to get obs: {}", e)))?;
    let var_df = parts
        .dollar("var")
        .map_err(|e| Error::Other(format!("failed to get var: {}", e)))?;

    // Transpose dgCMatrix (CSC, genes × cells) → CSR (cells × genes)
    let (csr_indptr, csr_indices, values_bytes, n_obs, n_vars, value_encoding) =
        dgcmatrix_to_csr(&counts)?;

    // Convert obs/var dataframes to RecordBatch
    let obs_batch = dataframe_to_record_batch(&obs_df)?;
    let var_batch = dataframe_to_record_batch(&var_df)?;

    // Write SCX file
    write_csr_to_scx(
        output_path,
        &csr_indptr,
        &csr_indices,
        &values_bytes,
        value_encoding,
        n_obs,
        n_vars,
        &obs_batch,
        &var_batch,
        explicit_codec,
        csc_always,
        csc_cols_per_shard,
    )
}

/// Import a SingleCellExperiment to an SCX file.
/// Same CSC→CSR transpose as from_seurat.
///
/// `csc` and `csc_cols_per_shard` mirror `from_seurat` — see those docs.
/// @export
#[extendr]
pub fn from_sce(
    sce_obj: Robj,
    output_path: &str,
    codec: Option<&str>,
    csc: Option<bool>,
    csc_cols_per_shard: Option<i32>,
) -> Result<()> {
    let explicit_codec = parse_codec_r(codec)?;
    let csc_always = csc.unwrap_or(false);
    let csc_cols_per_shard = csc_cols_per_shard
        .map(|v| {
            if v < 0 {
                Err(Error::Other(format!(
                    "csc_cols_per_shard must be >= 0, got {}",
                    v
                )))
            } else {
                Ok(v as usize)
            }
        })
        .transpose()?
        .unwrap_or(5000);
    // Single R!() call — moves sce_obj once, returns a lightweight list
    let parts = R!("
        if (!requireNamespace('SingleCellExperiment', quietly = TRUE))
            stop('SingleCellExperiment is required for from_sce()')
        sce <- {{sce_obj}}
        counts <- SummarizedExperiment::assay(sce, 'counts')
        obs_df <- as.data.frame(SummarizedExperiment::colData(sce))
        var_df <- as.data.frame(SummarizedExperiment::rowData(sce))
        list(counts = counts, obs = obs_df, var = var_df)
    ")
    .map_err(|e| Error::Other(format!("failed to extract SCE data: {}", e)))?;

    // Unpack via Robj::dollar() — no additional R!() calls or clones
    let counts = parts
        .dollar("counts")
        .map_err(|e| Error::Other(format!("failed to get counts: {}", e)))?;
    let obs_df = parts
        .dollar("obs")
        .map_err(|e| Error::Other(format!("failed to get obs: {}", e)))?;
    let var_df = parts
        .dollar("var")
        .map_err(|e| Error::Other(format!("failed to get var: {}", e)))?;

    // Transpose dgCMatrix (CSC, genes × cells) → CSR (cells × genes)
    let (csr_indptr, csr_indices, values_bytes, n_obs, n_vars, value_encoding) =
        dgcmatrix_to_csr(&counts)?;

    // Convert obs/var dataframes to RecordBatch
    let obs_batch = dataframe_to_record_batch(&obs_df)?;
    let var_batch = dataframe_to_record_batch(&var_df)?;

    // Write SCX file
    write_csr_to_scx(
        output_path,
        &csr_indptr,
        &csr_indices,
        &values_bytes,
        value_encoding,
        n_obs,
        n_vars,
        &obs_batch,
        &var_batch,
        explicit_codec,
        csc_always,
        csc_cols_per_shard,
    )
}

// ─── Module Registration ─────────────────────────────────────────────────────

extendr_module! {
    mod interop;
    fn from_seurat;
    fn from_sce;
}
