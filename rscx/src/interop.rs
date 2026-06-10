// Phase D: Seurat/SCE interop — CSR↔CSC, Arrow→data.frame
//
// This module provides the minimum viable interop functions needed by Phase B
// (reader.rs). Full Seurat/SCE conversion will be added in Phase D.

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::DataType;
use extendr_api::prelude::*;
use scx_codec::ValueEncoding;
use scx_format::ScxReader;
use scx_sparse::ScxCsr;

// ─── Arrow RecordBatch → R data.frame ────────────────────────────────────────

/// Canonical index-column names, in priority order. The engine emits
/// `__index_level_0__` for an unnamed pandas index (pyarrow convention);
/// `_index` is anndata's on-disk name; the rest are defensive fallbacks.
const INDEX_COLUMN_NAMES: &[&str] = &[
    "__index_level_0__",
    "_index",
    "index",
    "obs_names",
    "var_names",
];

/// Position of the canonical index column in `batch`, if any.
fn index_column_pos(batch: &RecordBatch) -> Option<usize> {
    let schema = batch.schema();
    INDEX_COLUMN_NAMES.iter().find_map(|&want| {
        schema
            .fields()
            .iter()
            .position(|f| f.name().as_str() == want)
    })
}

/// Extract a string-typed (`Utf8`/`LargeUtf8`) Arrow column as owned
/// `String`s (nulls → empty string). Returns `None` for non-string columns,
/// so callers fall back to synthetic names.
fn array_to_strings(col: &dyn Array) -> Option<Vec<String>> {
    match col.data_type() {
        DataType::Utf8 => {
            let a = col.as_string::<i32>();
            Some(
                (0..a.len())
                    .map(|i| {
                        if a.is_null(i) {
                            String::new()
                        } else {
                            a.value(i).to_string()
                        }
                    })
                    .collect(),
            )
        }
        DataType::LargeUtf8 => {
            let a = col.as_string::<i64>();
            Some(
                (0..a.len())
                    .map(|i| {
                        if a.is_null(i) {
                            String::new()
                        } else {
                            a.value(i).to_string()
                        }
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// The canonical index column's values as `String`s, if the batch carries
/// one and it is string-typed (barcodes for obs, gene IDs for var).
fn index_column_strings(batch: &RecordBatch) -> Option<Vec<String>> {
    let pos = index_column_pos(batch)?;
    array_to_strings(batch.column(pos))
}

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

    // B1: the canonical index column (barcodes / gene IDs) becomes
    // row.names rather than an ordinary column — but only when it is
    // string-typed and we can actually use it; otherwise leave it as a
    // regular column and fall back to synthetic row names so no data is lost.
    let row_names: Option<Vec<String>> =
        index_column_pos(batch).and_then(|pos| array_to_strings(batch.column(pos)));
    let skip_idx = if row_names.is_some() {
        index_column_pos(batch)
    } else {
        None
    };

    // Build a named list of R vectors, one per retained column
    let mut columns: Vec<(&str, Robj)> = Vec::with_capacity(n_cols);

    for (i, field) in schema.fields().iter().enumerate() {
        if Some(i) == skip_idx {
            continue;
        }
        let col = batch.column(i);
        let name = field.name().as_str();
        let ordered = field
            .metadata()
            .get(CATEGORICAL_ORDERED_KEY)
            .map(|v| v == "true")
            .unwrap_or(false);
        let robj = arrow_column_to_robj(col, field.data_type(), ordered)?;
        columns.push((name, robj));
    }

    // Convert named list to data.frame
    let list = List::from_pairs(columns);
    // Set class to "data.frame" and row.names
    let n_rows = batch.num_rows() as i32;
    let robj: Robj = list.into();

    // Preserve the index column as row.names so name-based joins and
    // Seurat's rowname-alignment work, instead of overwriting it with a
    // synthetic 1..N sequence.
    if let Some(names) = row_names {
        return R!(
            "{ x <- {{robj}}; class(x) <- 'data.frame'; attr(x, 'row.names') <- {{names}}; x }"
        )
        .map_err(|e| Error::Other(format!("data.frame construction failed: {}", e)));
    }

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
fn arrow_column_to_robj(col: &dyn Array, dtype: &DataType, ordered: bool) -> Result<Robj> {
    use arrow::datatypes::*;
    match dtype {
        // String types → character vector
        DataType::Utf8 => string_to_robj::<i32>(col),
        DataType::LargeUtf8 => string_to_robj::<i64>(col),

        // Dictionary → R factor (ordered factor when the field is marked so)
        DataType::Dictionary(key_type, value_type) => {
            dictionary_to_factor(col, key_type, value_type, ordered)
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
    ordered: bool,
) -> Result<Robj> {
    // We support Dictionary<Int8/16/32, Utf8> which is the common h5ad categorical pattern
    match (key_type, value_type) {
        (DataType::Int8, DataType::Utf8) => {
            typed_dict_to_factor::<arrow::datatypes::Int8Type>(col, ordered)
        }
        (DataType::Int16, DataType::Utf8) => {
            typed_dict_to_factor::<arrow::datatypes::Int16Type>(col, ordered)
        }
        (DataType::Int32, DataType::Utf8) => {
            typed_dict_to_factor::<arrow::datatypes::Int32Type>(col, ordered)
        }
        _ => Err(Error::Other(format!(
            "unsupported Dictionary key/value types: {:?}/{:?}",
            key_type, value_type
        ))),
    }
}

/// Helper: extract factor levels and codes from a typed DictionaryArray.
fn typed_dict_to_factor<K>(col: &dyn Array, ordered: bool) -> Result<Robj>
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

    // Extract codes (1-based for R factors, NA for nulls).
    // B7: bounds-check each key against the emitted levels. A key outside
    // `0..levels.len()` (e.g. from a sliced/non-canonical dictionary) would
    // otherwise render as a silent `<NA>` or index past the levels vector —
    // error explicitly instead.
    let levels_len = levels.len();
    let keys = dict_arr.keys();
    let codes: Vec<Option<i32>> = (0..keys.len())
        .map(|i| -> Result<Option<i32>> {
            if dict_arr.is_null(i) {
                Ok(None)
            } else {
                // Arrow indices are 0-based, R factor codes are 1-based.
                let key_val = keys.as_primitive::<K>().value(i);
                let idx: i32 = key_val.try_into().map_err(|_| {
                    Error::Other(format!("dictionary key at row {i} does not fit in i32"))
                })?;
                if idx < 0 || (idx as usize) >= levels_len {
                    return Err(Error::Other(format!(
                        "dictionary key {idx} at row {i} is out of range for {levels_len} levels"
                    )));
                }
                Ok(Some(idx + 1)) // 0-based → 1-based
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // Build factor in R: integer vector with "levels" and "class" attributes.
    // An ordered categorical (marked via the field metadata) becomes an
    // ordered factor (`class = c("ordered", "factor")`) so the bit round-trips.
    let levels_robj: Robj = levels.into_robj();
    let codes_robj: Robj = codes.into_robj();
    let class_robj: Robj = if ordered {
        vec!["ordered", "factor"].into_robj()
    } else {
        "factor".into_robj()
    };
    R!("{ x <- {{codes_robj}}; attr(x, 'levels') <- {{levels_robj}}; class(x) <- {{class_robj}}; x }")
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

/// Set a cells×genes dgCMatrix's `Dimnames` from the obs/var index columns
/// (rows = cell barcodes, cols = gene IDs) so the resulting Seurat/SCE object
/// carries real names (B1). When either index is missing, the matrix is
/// returned unchanged (Dimnames stays NULL, the prior behaviour).
fn set_dgc_dimnames(dgc: Robj, obs: &RecordBatch, var: &RecordBatch) -> Result<Robj> {
    match (index_column_strings(obs), index_column_strings(var)) {
        (Some(cells), Some(genes)) => {
            R!("{ m <- {{dgc}}; dimnames(m) <- list({{cells}}, {{genes}}); m }")
                .map_err(|e| Error::Other(format!("setting dgCMatrix dimnames failed: {}", e)))
        }
        _ => Ok(dgc),
    }
}

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
    let dgc = set_dgc_dimnames(dgc, &result.obs, &result.var)?;
    let obs_df = record_batch_to_dataframe(&result.obs)?;
    let var_df = record_batch_to_dataframe(&result.var)?;

    R!("
        if (!requireNamespace('Seurat', quietly = TRUE))
            stop('Seurat >= 5.0.0 is required for to_seurat()')
        counts_t <- Matrix::t({{dgc}})
        seu <- Seurat::CreateSeuratObject(counts = counts_t)
        # Attach obs via AddMetaData keyed on barcodes: preserves Seurat's
        # computed columns (nCount_RNA, nFeature_RNA) that a wholesale
        # `seu@meta.data <-` would wipe, and survives any CreateSeuratObject
        # cell filtering. Align by barcode when present, else positionally.
        md <- {{obs_df}}
        if (all(colnames(seu) %in% rownames(md))) {
            md <- md[colnames(seu), , drop = FALSE]
        } else if (nrow(md) == ncol(seu)) {
            rownames(md) <- colnames(seu)
        } else {
            stop('to_seurat: obs metadata rows do not align to Seurat cells')
        }
        seu <- Seurat::AddMetaData(seu, metadata = md)
        # Feature metadata: align to the assay's feature order by name when
        # gene IDs are present, else attach positionally.
        vd <- {{var_df}}
        if (all(rownames(seu) %in% rownames(vd))) {
            vd <- vd[rownames(seu), , drop = FALSE]
        }
        # 'RNA' is intentional here: this write path always builds the assay as
        # 'RNA' via CreateSeuratObject(counts=) above. The T4.2 de-hardcoding of
        # the assay name targets the from_seurat *read* path, not this writer.
        seu[['RNA']]@meta.data <- vd
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
    let dgc = set_dgc_dimnames(dgc, &result.obs, &result.var)?;
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

    // Detect value encoding via the shared scx-codec policy (I-ORG-1 / T4.9 —
    // the `f64` variant avoids a lossy f64→f32 round-trip on the raw-Census
    // count path). The byte serialization stays f64-native here so values
    // beyond f32's 2^24 contiguous-integer range keep full precision.
    let value_encoding = scx_codec::detect_value_encoding_f64(&csr_values_f64);
    let values_bytes: Vec<u8> = match value_encoding {
        ValueEncoding::Uint8 => csr_values_f64.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(csr_values_f64.len() * 2);
            for &v in &csr_values_f64 {
                buf.extend_from_slice(&(v as u16).to_le_bytes());
            }
            buf
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(csr_values_f64.len() * 4);
            for &v in &csr_values_f64 {
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
            buf
        }
        _ => {
            let mut buf = Vec::with_capacity(csr_values_f64.len() * 4);
            for &v in &csr_values_f64 {
                buf.extend_from_slice(&(v as f32).to_le_bytes());
            }
            buf
        }
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
/// Field-metadata key marking an Arrow dictionary column as an *ordered*
/// categorical, sourced from the canonical `scx-format` definition (shared by
/// scx-convert and pyscx too) so the wire key has a single source of truth.
use scx_format::CATEGORICAL_ORDERED_KEY;

fn dataframe_to_record_batch(df: &Robj) -> Result<arrow::array::RecordBatch> {
    use arrow::array::{BooleanArray, DictionaryArray, Float64Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let names_robj =
        R!("names({{df}})").map_err(|e| Error::Other(format!("failed to get names: {}", e)))?;
    let col_names: Vec<String> = names_robj
        .as_str_iter()
        .ok_or_else(|| Error::Other("names not character".into()))?
        .map(|s| s.to_string())
        .collect();

    // Row count must be captured up front: a 0-column data.frame (e.g. a
    // Seurat object with no feature metadata) yields an empty schema, and
    // `RecordBatch::try_new` cannot infer the row count from zero arrays.
    let n_rows = R!("nrow({{df}})")
        .map_err(|e| Error::Other(format!("nrow() failed: {}", e)))?
        .as_integer()
        .ok_or_else(|| Error::Other("nrow() not a scalar integer".into()))?
        as usize;

    let n_cols = col_names.len();
    let mut fields = Vec::with_capacity(n_cols);
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(n_cols);

    // A data.frame is a VECSXP (list of columns); pull the columns once as a
    // list so each column access is a Rust vector index rather than a runtime
    // `R!("df[[i]]")` evaluation.
    let columns: Vec<Robj> = df
        .as_list()
        .ok_or_else(|| Error::Other("data.frame is not a list".into()))?
        .values()
        .collect();

    for (i, name) in col_names.iter().enumerate() {
        let col_robj = &columns[i];

        // Detect the R class to map to the closest Arrow type (T4.3), so
        // meta.data column classes round-trip via the reverse map in
        // `arrow_column_to_robj`: factor → Dictionary(Int32,Utf8) (carrying
        // the `ordered` bit), integer → Int32, logical → Boolean, double →
        // Float64, character → Utf8. Detection uses extendr's native
        // `rtype()`/`inherits()` rather than per-column `R!` evaluations (which
        // parse + eval R at runtime — a real cost across many columns). A
        // factor is checked first because it is an INTSXP underneath.
        let is_factor = col_robj.inherits("factor");
        let is_character = col_robj.rtype() == Rtype::Strings;
        let is_logical = col_robj.rtype() == Rtype::Logicals;
        let is_integer = col_robj.rtype() == Rtype::Integers && !is_factor;

        if is_factor {
            // Factor → Arrow Dictionary(Int32, Utf8): levels become the
            // dictionary, R's 1-based codes (NA → null) become 0-based keys.
            let levels: Vec<String> = col_robj
                .levels()
                .ok_or_else(|| Error::Other(format!("levels({name}) missing/not character")))?
                .map(|s| s.to_string())
                .collect();
            // A factor's underlying storage is its 1-based integer codes.
            let keys: Int32Array = col_robj
                .as_integer_slice()
                .ok_or_else(|| Error::Other(format!("factor codes for {name} not integer")))?
                .iter()
                .map(|&c| if c == i32::MIN { None } else { Some(c - 1) }) // NA sentinel; 1- → 0-based
                .collect();
            let values: arrow::array::ArrayRef = Arc::new(StringArray::from(
                levels.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ));
            let dict = DictionaryArray::<Int32Type>::try_new(keys, values)
                .map_err(|e| Error::Other(format!("dictionary array for {name}: {e}")))?;

            let is_ordered = col_robj.inherits("ordered");
            let mut field = Field::new(
                name,
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            );
            if is_ordered {
                let mut md = HashMap::new();
                md.insert(CATEGORICAL_ORDERED_KEY.to_string(), "true".to_string());
                field = field.with_metadata(md);
            }
            fields.push(field);
            arrays.push(Arc::new(dict));
        } else if is_character {
            let strings: Vec<Option<String>> = col_robj
                .as_str_iter()
                .ok_or_else(|| Error::Other(format!("column {} not iterable as str", name)))?
                .map(|s| Some(s.to_string()))
                .collect();
            let arr = StringArray::from(strings.iter().map(|s| s.as_deref()).collect::<Vec<_>>());
            fields.push(Field::new(name, DataType::Utf8, true));
            arrays.push(Arc::new(arr));
        } else if is_logical {
            // R logical → Arrow Boolean (NA → null). R NA is the i32::MIN
            // sentinel in the logical slice.
            let arr: BooleanArray = col_robj
                .as_logical_slice()
                .ok_or_else(|| Error::Other(format!("column {} not logical", name)))?
                .iter()
                .map(|b| if b.is_na() { None } else { Some(b.is_true()) })
                .collect();
            fields.push(Field::new(name, DataType::Boolean, true));
            arrays.push(Arc::new(arr));
        } else if is_integer {
            // R integer → Arrow Int32 (NA → null via the i32::MIN sentinel).
            let arr: Int32Array = col_robj
                .as_integer_slice()
                .ok_or_else(|| Error::Other(format!("column {} not integer", name)))?
                .iter()
                .map(|&v| if v == i32::MIN { None } else { Some(v) })
                .collect();
            fields.push(Field::new(name, DataType::Int32, true));
            arrays.push(Arc::new(arr));
        } else {
            // double (and any other numeric) → Float64 (NaN → null).
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

    // Preserve the data.frame's explicit row names (cell barcodes / gene IDs)
    // as the canonical `__index_level_0__` index column so they round-trip;
    // `record_batch_to_dataframe` reads it back into `row.names`. R stores
    // auto-generated row names as an integer vector (`1..n`) and explicit ones
    // as a character vector, so emit only the latter.
    let row_names: Vec<String> = R!("if (is.character(attr({{df}}, 'row.names'))) \
            as.character(attr({{df}}, 'row.names')) else character(0)")
    .map_err(|e| Error::Other(format!("row.names extraction failed: {}", e)))?
    .as_str_vector()
    .unwrap_or_default()
    .iter()
    .map(|s| s.to_string())
    .collect();
    if row_names.len() == n_rows {
        let arr = StringArray::from(row_names.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        fields.push(Field::new("__index_level_0__", DataType::Utf8, false));
        arrays.push(Arc::new(arr));
    }

    let schema = Arc::new(Schema::new(fields));
    // Use the explicit-row-count constructor so a 0-column metadata frame
    // (no obs/var columns) still produces a valid n_rows-row batch.
    let options = arrow::array::RecordBatchOptions::new().with_row_count(Some(n_rows));
    arrow::array::RecordBatch::try_new_with_options(schema, arrays, &options)
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
    use scx_format::writer::ScxWriter;
    use scx_format::{select_codec_for_modality, ModalityType};

    let nnz = *csr_indptr.last().unwrap_or(&0);
    let shard_target_rows: usize = 16384;
    let n_shards = n_obs.div_ceil(shard_target_rows.max(1));

    let header = FileHeader {
        n_obs: n_obs as u64,
        n_vars: n_vars as u64,
        nnz,
        n_csr_shards: n_shards as u32,
        shard_target_rows: shard_target_rows as u32,
        codec_id: CodecId::None as u8,
        index_dtype: if n_vars <= 65535 { 0 } else { 1 },
        manifest_sequence: 1,
        ..Default::default()
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
            None => select_codec_for_modality(shard_values, value_encoding, ModalityType::Rna),
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
/// (4 GiB, matches scx-convert and pyscx).
const RSCX_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Streaming CSR → CSC transpose over the in-memory `(csr_indptr,
/// csr_indices, values_bytes)` arrays, writing each chunk as one CSC
/// shard. Decodes the pre-encoded value bytes to f32, then delegates to
/// the shared `scx_format::csc_sidecar::write_csc_sidecar` so this and the
/// `scx-convert` / `pyscx` import paths produce structurally identical CSC
/// sidecars (I-ORG-1 / T4.9 — single transpose-and-write loop).
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
    scx_format::csc_sidecar::write_csc_sidecar(
        writer,
        std::slice::from_ref(&csr),
        n_obs,
        n_vars,
        value_encoding,
        codec_id,
        csc_cols_per_shard,
        RSCX_CSC_MEMORY_BYTES,
        None,
    )
    .map_err(|e| Error::Other(format!("CSC sidecar write failed: {}", e)))
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

    // Phase I.1: detect Seurat v5 multi-assay objects. If the object
    // has more than one non-empty assay, route to the multimodal
    // write path so each assay lands as its own modality in the SCX
    // file. Single-assay objects keep the legacy single-modality
    // path below.
    let assay_names_vec: Vec<String> = {
        let assay_names = R!("
            if (!requireNamespace('SeuratObject', quietly = TRUE) &&
                !requireNamespace('Seurat', quietly = TRUE))
                stop('Seurat >= 5.0.0 is required for from_seurat()')
            seu <- {{&seurat_obj}}
            names(seu@assays)
        ")
        .map_err(|e| Error::Other(format!("failed to read Seurat assay names: {}", e)))?;
        assay_names
            .as_str_vector()
            .ok_or_else(|| Error::Other("assay names not a character vector".into()))?
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    if assay_names_vec.len() > 1 {
        return from_seurat_multi_assay(
            seurat_obj,
            &assay_names_vec,
            output_path,
            explicit_codec,
            csc_always,
            csc_cols_per_shard,
        );
    }

    // The sole assay's actual name — do not hardcode 'RNA', so an object
    // whose only assay is SCT / originalexp / etc. round-trips its feature
    // metadata. (At this point `assay_names_vec.len() <= 1`.)
    let sole_assay = assay_names_vec.first().map(|s| s.as_str()).unwrap_or("RNA");

    // Single R!() call — moves seurat_obj once, returns a lightweight list
    let parts = R!("
        if (!requireNamespace('Seurat', quietly = TRUE))
            stop('Seurat >= 5.0.0 is required for from_seurat()')
        seu <- {{seurat_obj}}
        counts <- Seurat::GetAssayData(seu, layer = 'counts')
        obs_df <- seu@meta.data
        var_df <- tryCatch(seu[[{{sole_assay}}]]@meta.data,
            error = function(e) data.frame(gene_id = rownames(seu)))
        # An assay's feature metadata can come back with no (or auto-integer)
        # row names; pin them to the feature names so gene IDs round-trip and
        # to_seurat() can restore dimnames.
        if (nrow(var_df) == length(rownames(seu))) rownames(var_df) <- rownames(seu)
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

// ─── Phase I.1: Seurat v5 multi-assay path ──────────────────────────────────

/// Phase I.1: write a Seurat v5 multi-assay object as a multimodal
/// SCX file. Cells must align across assays (Seurat v5's invariant);
/// per-assay counts are stamped with their own `modality_id`.
fn from_seurat_multi_assay(
    seurat_obj: Robj,
    assay_names: &[String],
    output_path: &str,
    explicit_codec: Option<scx_codec::CodecId>,
    csc_always: bool,
    csc_cols_per_shard: usize,
) -> Result<()> {
    use scx_codec::CodecId;
    use scx_format::header::FileHeader;
    use scx_format::writer::ScxWriter;
    use scx_format::{select_codec_for_modality, ModalityType};

    if csc_always {
        return Err(Error::Other(
            "from_seurat(csc=TRUE) on a multi-assay object is not yet supported; \
             write per-modality CSC sidecars via `scx build-csc` after import"
                .into(),
        ));
    }
    let _ = csc_cols_per_shard; // unused on the no-CSC path

    // Pull the global obs (meta.data) once. Seurat shares meta.data
    // across assays, so it lives at modality_id = 0 (global) in SCX.
    let obs_robj = R!("{{&seurat_obj}}@meta.data")
        .map_err(|e| Error::Other(format!("failed to read meta.data: {}", e)))?;
    let obs_batch = dataframe_to_record_batch(&obs_robj)?;

    // Per-assay extraction: counts (dgCMatrix, genes × cells) + var
    // (feature metadata data.frame). We pull these in one R!() per
    // assay to minimise marshalling overhead.
    struct AssayPayload {
        name: String,
        n_obs: usize,
        n_vars: usize,
        csr_indptr: Vec<u64>,
        csr_indices: Vec<u32>,
        values_bytes: Vec<u8>,
        value_encoding: ValueEncoding,
        var_batch: RecordBatch,
        modality_type: ModalityType,
    }
    let mut payloads: Vec<AssayPayload> = Vec::with_capacity(assay_names.len());
    let mut shared_n_obs: Option<usize> = None;
    for name in assay_names {
        let parts = R!("
            seu <- {{&seurat_obj}}
            assay_name <- {{name.as_str()}}
            counts <- Seurat::GetAssayData(seu, assay = assay_name, layer = 'counts')
            var_df <- tryCatch(seu[[assay_name]]@meta.data,
                error = function(e) data.frame(feature_id = rownames(counts)))
            list(counts = counts, var = var_df)
        ")
        .map_err(|e| Error::Other(format!("failed to extract assay '{name}': {e}")))?;
        let counts = parts
            .dollar("counts")
            .map_err(|e| Error::Other(format!("failed to get counts for '{name}': {e}")))?;
        let var_df = parts
            .dollar("var")
            .map_err(|e| Error::Other(format!("failed to get var for '{name}': {e}")))?;

        let (csr_indptr, csr_indices, values_bytes, n_obs, n_vars, value_encoding) =
            dgcmatrix_to_csr(&counts)?;
        let var_batch = dataframe_to_record_batch(&var_df)?;
        let modality_type = infer_modality_type_from_name(name);

        // Cell-axis alignment check across assays.
        match shared_n_obs {
            None => shared_n_obs = Some(n_obs),
            Some(expected) if expected == n_obs => {}
            Some(expected) => {
                return Err(Error::Other(format!(
                    "from_seurat: assay '{name}' has n_obs = {n_obs} but the \
                     first assay had n_obs = {expected}. Seurat v5 multi-assay \
                     objects require identical cell axes across all assays."
                )));
            }
        }

        payloads.push(AssayPayload {
            name: name.clone(),
            n_obs,
            n_vars,
            csr_indptr,
            csr_indices,
            values_bytes,
            value_encoding,
            var_batch,
            modality_type,
        });
    }

    let n_obs = shared_n_obs.unwrap_or(0);
    let max_n_vars = payloads.iter().map(|p| p.n_vars).max().unwrap_or(0) as u64;

    // Build the multimodal v2 header. n_csr_shards / n_csc_shards /
    // modality_table_offset are filled in by ScxWriter::finish based
    // on the registered modalities + write_csr_shard_for calls.
    let header = FileHeader::new_single_modality(
        n_obs as u64,
        max_n_vars,
        0,
        16384,
        CodecId::None as u8,
        if max_n_vars <= 65535 { 0 } else { 1 },
    );
    let mut writer = ScxWriter::new(output_path, header)
        .map_err(|e| Error::Other(format!("ScxWriter::new failed: {}", e)))?;

    writer
        .write_obs(&obs_batch)
        .map_err(|e| Error::Other(format!("write_obs failed: {}", e)))?;

    // Per-modality writes: register modality, write var, write CSR
    // shards in shard_target_rows row chunks.
    let shard_target_rows: usize = 16384;
    for payload in &payloads {
        // Resolve the per-modality auto-codec by feeding the first
        // shard's bytes through `select_codec_for_modality`. The
        // modality table records this codec as the default.
        let resolved_codec = match explicit_codec {
            Some(c) => {
                if c == CodecId::Scx1 && !payload.value_encoding.is_integer() {
                    CodecId::Zstd
                } else {
                    c
                }
            }
            None => select_codec_for_modality(
                &payload.values_bytes,
                payload.value_encoding,
                payload.modality_type,
            ),
        };

        let modality_id = writer
            .add_modality(
                &payload.name,
                payload.modality_type,
                resolved_codec,
                payload.value_encoding,
                false, // build_csc: from_seurat / from_mae's csc=TRUE path
                       // is rejected explicitly above, so no auto-emit.
            )
            .map_err(|e| Error::Other(format!("add_modality({}) failed: {}", payload.name, e)))?;
        writer
            .set_modality_n_vars(modality_id, payload.n_vars as u64)
            .map_err(|e| Error::Other(format!("set_modality_n_vars failed: {}", e)))?;
        writer
            .write_var_for(modality_id, &payload.var_batch)
            .map_err(|e| Error::Other(format!("write_var_for failed: {}", e)))?;

        let bw = payload.value_encoding.byte_width();
        let mut row_start: usize = 0;
        while row_start < payload.n_obs {
            let row_end = (row_start + shard_target_rows).min(payload.n_obs);
            let base = payload.csr_indptr[row_start];
            let shard_indptr: Vec<u64> = payload.csr_indptr[row_start..=row_end]
                .iter()
                .map(|&v| v - base)
                .collect();
            let nnz_start = base as usize;
            let nnz_end = payload.csr_indptr[row_end] as usize;
            let shard_indices: Vec<u32> = payload.csr_indices[nnz_start..nnz_end].to_vec();
            let shard_values = &payload.values_bytes[nnz_start * bw..nnz_end * bw];
            writer
                .write_csr_shard_for(
                    modality_id,
                    &shard_indptr,
                    &shard_indices,
                    shard_values,
                    resolved_codec,
                    payload.value_encoding,
                    row_start as u64,
                )
                .map_err(|e| Error::Other(format!("write_csr_shard_for failed: {}", e)))?;
            row_start = row_end;
        }
    }

    writer
        .finish()
        .map_err(|e| Error::Other(format!("finish failed: {}", e)))?;
    Ok(())
}

/// Phase I.1 / I.2: best-effort modality-type inference from an assay
/// name. Mirrors the heuristic in `pyscx::mudata::infer_modality_type`
/// so multimodal SCX files written from Seurat / MAE / MuData carry
/// consistent `ModalityType` tags.
fn infer_modality_type_from_name(name: &str) -> scx_format::ModalityType {
    let lower = name.to_ascii_lowercase();
    if lower.contains("atac") || lower.contains("peak") {
        scx_format::ModalityType::Atac
    } else if lower.contains("adt") || lower.contains("protein") || lower.contains("antibody") {
        scx_format::ModalityType::Protein
    } else if lower.contains("spatial") {
        scx_format::ModalityType::Spatial
    } else if lower.contains("methyl") {
        scx_format::ModalityType::Methylation
    } else if lower == "rna" || lower == "gex" || lower.contains("expression") {
        scx_format::ModalityType::Rna
    } else {
        scx_format::ModalityType::Custom
    }
}

// ─── Phase I.1: multimodal to_seurat / Phase I.2: MAE bindings ──────────────

/// Phase I.1: build a Seurat v5 multi-assay object from a multimodal
/// SCX reader. One `Assay5` per registered modality; meta.data and
/// cell names come from the global obs.
pub fn to_seurat_multimodal(reader: &ScxReader) -> Result<Robj> {
    if !reader.is_multimodal() {
        return Err(Error::Other(
            "to_seurat: source file is single-modality; use the per-result \
             to_seurat() method instead"
                .into(),
        ));
    }
    let modality_names: Vec<String> = reader
        .modality_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let obs_batch = reader
        .read_obs()
        .map_err(|e| Error::Other(format!("read_obs failed: {}", e)))?;
    let obs_df = record_batch_to_dataframe(&obs_batch)?;

    // Build the v5 object incrementally. We seed it with the first
    // modality (Seurat v5 requires at least one assay at construction
    // time), then attach the rest via seu[[name]] <- assay.
    let n_modalities = modality_names.len();
    if n_modalities == 0 {
        return Err(Error::Other(
            "to_seurat: file is multimodal but the modality table is empty".into(),
        ));
    }

    // Helper closure: turn a (modality_name, modality_id) pair into
    // a Seurat Assay5 robj on the R side.
    let mut assay_robjs: Vec<(String, Robj)> = Vec::with_capacity(n_modalities);
    for name in &modality_names {
        let mid = reader.modality_id(name).ok_or_else(|| {
            Error::Other(format!("modality '{name}' missing from modality table"))
        })?;
        let csr = reader
            .read_all_csr_shards_for(mid)
            .map_err(|e| Error::Other(format!("read_all_csr_shards_for({name}): {}", e)))?;
        let dgc = csr_to_dgcmatrix(&csr)?;
        let var_batch = reader
            .read_var_for(mid)
            .map_err(|e| Error::Other(format!("read_var_for({name}): {}", e)))?;
        let var_df = record_batch_to_dataframe(&var_batch)?;
        // Build an Assay5 from the per-modality counts + features.
        let assay = R!("
            if (!requireNamespace('Seurat', quietly = TRUE))
                stop('Seurat >= 5.0.0 is required for to_seurat()')
            counts_t <- Matrix::t({{dgc}})
            a <- SeuratObject::CreateAssay5Object(counts = counts_t)
            features_df <- {{var_df}}
            if (nrow(features_df) == nrow(a)) {
                # Attach feature metadata to the assay's @meta.data slot.
                a@meta.data <- features_df
            }
            a
        ")
        .map_err(|e| Error::Other(format!("CreateAssay5Object({name}): {e}")))?;
        assay_robjs.push((name.clone(), assay));
    }

    // Assemble the Seurat v5 object: seed with the first modality,
    // then add the rest via `seu[[name]] <- assay`. The first
    // modality's name is used as the default assay (matches Seurat's
    // single-assay convention).
    let (first_name, first_assay) = &assay_robjs[0];
    let seu = R!("
        seu <- Seurat::CreateSeuratObject(counts = {{first_assay}}, assay = {{first_name.as_str()}})
        # AddMetaData keyed on barcodes (see to_seurat_v5): preserves the
        # per-assay computed columns and survives any cell filtering.
        md <- {{obs_df}}
        if (all(colnames(seu) %in% rownames(md))) {
            md <- md[colnames(seu), , drop = FALSE]
        } else if (nrow(md) == ncol(seu)) {
            rownames(md) <- colnames(seu)
        } else {
            stop('to_seurat: obs metadata rows do not align to Seurat cells')
        }
        seu <- Seurat::AddMetaData(seu, metadata = md)
        seu
    ")
    .map_err(|e| Error::Other(format!("CreateSeuratObject: {e}")))?;

    let final_seu = if assay_robjs.len() > 1 {
        let mut current = seu;
        for (name, assay) in assay_robjs.iter().skip(1) {
            current = R!("
                seu <- {{current}}
                seu[[{{name.as_str()}}]] <- {{assay}}
                seu
            ")
            .map_err(|e| Error::Other(format!("attach assay '{name}': {e}")))?;
        }
        current
    } else {
        seu
    };
    Ok(final_seu)
}

/// Phase I.2: import a Bioconductor MultiAssayExperiment to a
/// multimodal SCX file. Requires that all experiments share the same
/// cell axis (`colnames` aligned across experiments). On
/// misalignment, raises a clear error directing the user to align
/// upfront via `intersectColumns()` or similar.
#[extendr]
pub fn from_mae(
    mae_obj: Robj,
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

    // Pull experiment names + per-experiment cell-axis fingerprints
    // in a single R!() so we can validate alignment before any
    // SCX-side allocation.
    let info = R!("
        if (!requireNamespace('MultiAssayExperiment', quietly = TRUE))
            stop('MultiAssayExperiment is required for from_mae()')
        mae <- {{&mae_obj}}
        exps <- MultiAssayExperiment::experiments(mae)
        nm <- names(exps)
        # Collect colnames per experiment for alignment check.
        col_lists <- lapply(seq_along(exps), function(i) colnames(exps[[i]]))
        # Reference: union of cells from sampleMap (canonical for MAE).
        ref_cells <- colnames(MultiAssayExperiment::colData(mae))
        if (is.null(ref_cells)) ref_cells <- rownames(MultiAssayExperiment::colData(mae))
        list(names = nm, col_lists = col_lists, ref_cells = ref_cells)
    ")
    .map_err(|e| Error::Other(format!("failed to inspect MAE: {}", e)))?;

    let names_robj = info
        .dollar("names")
        .map_err(|e| Error::Other(format!("MAE names: {e}")))?;
    let assay_names: Vec<String> = names_robj
        .as_str_vector()
        .ok_or_else(|| Error::Other("MAE experiment names missing".into()))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    if assay_names.is_empty() {
        return Err(Error::Other("from_mae: MAE has no experiments".into()));
    }

    // Per-experiment cell-alignment check. We compare each
    // experiment's colnames against the first experiment's; on
    // mismatch, raise.
    let aligned = R!("
        col_lists <- {{info.dollar(\"col_lists\").map_err(|e| Error::Other(format!(\"{e}\")))?}}
        if (length(col_lists) <= 1) return(TRUE)
        ref <- col_lists[[1]]
        for (i in seq_along(col_lists)[-1]) {
            if (!identical(col_lists[[i]], ref)) return(FALSE)
        }
        TRUE
    ")
    .map_err(|e| Error::Other(format!("MAE alignment check: {e}")))?;
    let aligned_b: bool = aligned.as_logical().map(|b| b.is_true()).unwrap_or(false);
    if !aligned_b {
        return Err(Error::Other(
            "from_mae: MAE experiments have non-aligned cell axes (different \
             colnames across experiments). SCX requires a shared obs axis. \
             Use MultiAssayExperiment::intersectColumns(mae) to keep only the \
             cells present in every experiment, or NA-pad upfront, before \
             calling from_mae()."
                .into(),
        ));
    }

    // Now extract per-experiment counts as dgCMatrix and per-experiment
    // feature metadata. We reuse from_seurat_multi_assay's per-payload
    // structure by routing through a shared multi-modality writer.
    if csc_always {
        return Err(Error::Other(
            "from_mae(csc=TRUE) on a multi-experiment MAE is not yet supported; \
             use `scx build-csc` after import"
                .into(),
        ));
    }
    let _ = csc_cols_per_shard;

    use scx_codec::CodecId;
    use scx_format::header::FileHeader;
    use scx_format::writer::ScxWriter;
    use scx_format::{select_codec_for_modality, ModalityType};

    // colData becomes the global obs; rowData(experiments[[i]])
    // becomes per-modality var.
    let obs_df = R!("
        as.data.frame(MultiAssayExperiment::colData({{&mae_obj}}))
    ")
    .map_err(|e| Error::Other(format!("MAE colData: {e}")))?;
    let obs_batch = dataframe_to_record_batch(&obs_df)?;

    struct AssayPayload {
        name: String,
        n_obs: usize,
        n_vars: usize,
        csr_indptr: Vec<u64>,
        csr_indices: Vec<u32>,
        values_bytes: Vec<u8>,
        value_encoding: ValueEncoding,
        var_batch: RecordBatch,
        modality_type: ModalityType,
    }
    let mut payloads: Vec<AssayPayload> = Vec::with_capacity(assay_names.len());
    let mut shared_n_obs: Option<usize> = None;
    for name in &assay_names {
        let parts = R!("
            mae <- {{&mae_obj}}
            ename <- {{name.as_str()}}
            exp <- MultiAssayExperiment::experiments(mae)[[ename]]
            counts <- as(SummarizedExperiment::assay(exp), 'dgCMatrix')
            var_df <- as.data.frame(SummarizedExperiment::rowData(exp))
            list(counts = counts, var = var_df)
        ")
        .map_err(|e| Error::Other(format!("extract MAE experiment '{name}': {e}")))?;
        let counts = parts
            .dollar("counts")
            .map_err(|e| Error::Other(format!("MAE counts for '{name}': {e}")))?;
        let var_df = parts
            .dollar("var")
            .map_err(|e| Error::Other(format!("MAE var for '{name}': {e}")))?;
        let (csr_indptr, csr_indices, values_bytes, n_obs, n_vars, value_encoding) =
            dgcmatrix_to_csr(&counts)?;
        let var_batch = dataframe_to_record_batch(&var_df)?;
        let modality_type = infer_modality_type_from_name(name);
        match shared_n_obs {
            None => shared_n_obs = Some(n_obs),
            Some(expected) if expected == n_obs => {}
            Some(expected) => {
                return Err(Error::Other(format!(
                    "from_mae: experiment '{name}' has n_obs = {n_obs} but \
                     the first experiment had n_obs = {expected} after \
                     alignment. This indicates a corrupted MAE."
                )));
            }
        }
        payloads.push(AssayPayload {
            name: name.clone(),
            n_obs,
            n_vars,
            csr_indptr,
            csr_indices,
            values_bytes,
            value_encoding,
            var_batch,
            modality_type,
        });
    }

    let n_obs = shared_n_obs.unwrap_or(0);
    let max_n_vars = payloads.iter().map(|p| p.n_vars).max().unwrap_or(0) as u64;
    let header = FileHeader::new_single_modality(
        n_obs as u64,
        max_n_vars,
        0,
        16384,
        CodecId::None as u8,
        if max_n_vars <= 65535 { 0 } else { 1 },
    );
    let mut writer = ScxWriter::new(output_path, header)
        .map_err(|e| Error::Other(format!("ScxWriter::new failed: {}", e)))?;
    writer
        .write_obs(&obs_batch)
        .map_err(|e| Error::Other(format!("write_obs failed: {}", e)))?;

    let shard_target_rows: usize = 16384;
    for payload in &payloads {
        let resolved_codec = match explicit_codec {
            Some(c) => {
                if c == CodecId::Scx1 && !payload.value_encoding.is_integer() {
                    CodecId::Zstd
                } else {
                    c
                }
            }
            None => select_codec_for_modality(
                &payload.values_bytes,
                payload.value_encoding,
                payload.modality_type,
            ),
        };
        let modality_id = writer
            .add_modality(
                &payload.name,
                payload.modality_type,
                resolved_codec,
                payload.value_encoding,
                false, // build_csc: from_seurat / from_mae's csc=TRUE path
                       // is rejected explicitly above, so no auto-emit.
            )
            .map_err(|e| Error::Other(format!("add_modality({}) failed: {}", payload.name, e)))?;
        writer
            .set_modality_n_vars(modality_id, payload.n_vars as u64)
            .map_err(|e| Error::Other(format!("set_modality_n_vars failed: {}", e)))?;
        writer
            .write_var_for(modality_id, &payload.var_batch)
            .map_err(|e| Error::Other(format!("write_var_for failed: {}", e)))?;

        let bw = payload.value_encoding.byte_width();
        let mut row_start: usize = 0;
        while row_start < payload.n_obs {
            let row_end = (row_start + shard_target_rows).min(payload.n_obs);
            let base = payload.csr_indptr[row_start];
            let shard_indptr: Vec<u64> = payload.csr_indptr[row_start..=row_end]
                .iter()
                .map(|&v| v - base)
                .collect();
            let nnz_start = base as usize;
            let nnz_end = payload.csr_indptr[row_end] as usize;
            let shard_indices: Vec<u32> = payload.csr_indices[nnz_start..nnz_end].to_vec();
            let shard_values = &payload.values_bytes[nnz_start * bw..nnz_end * bw];
            writer
                .write_csr_shard_for(
                    modality_id,
                    &shard_indptr,
                    &shard_indices,
                    shard_values,
                    resolved_codec,
                    payload.value_encoding,
                    row_start as u64,
                )
                .map_err(|e| Error::Other(format!("write_csr_shard_for failed: {}", e)))?;
            row_start = row_end;
        }
    }

    writer
        .finish()
        .map_err(|e| Error::Other(format!("finish failed: {}", e)))?;
    Ok(())
}

/// Phase I.2: build a Bioconductor `MultiAssayExperiment` from a
/// multimodal SCX reader. Each modality becomes a
/// `SingleCellExperiment` in `experiments`; the global obs becomes
/// `colData`. Cell names are taken from the obs row index (or
/// auto-generated as `cell_0..n`).
pub fn to_mae(reader: &ScxReader) -> Result<Robj> {
    if !reader.is_multimodal() {
        return Err(Error::Other(
            "to_mae: source file is single-modality; use to_sce() instead".into(),
        ));
    }
    let modality_names: Vec<String> = reader
        .modality_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    if modality_names.is_empty() {
        return Err(Error::Other(
            "to_mae: file is multimodal but the modality table is empty".into(),
        ));
    }
    let obs_batch = reader
        .read_obs()
        .map_err(|e| Error::Other(format!("read_obs failed: {}", e)))?;
    let obs_df = record_batch_to_dataframe(&obs_batch)?;

    // Build a list of SingleCellExperiments (one per modality) on
    // the R side, then assemble a MAE.
    let mut sce_pairs: Vec<(String, Robj)> = Vec::with_capacity(modality_names.len());
    for name in &modality_names {
        let mid = reader.modality_id(name).ok_or_else(|| {
            Error::Other(format!("modality '{name}' missing from modality table"))
        })?;
        let csr = reader
            .read_all_csr_shards_for(mid)
            .map_err(|e| Error::Other(format!("read_all_csr_shards_for({name}): {e}")))?;
        let dgc = csr_to_dgcmatrix(&csr)?;
        let var_batch = reader
            .read_var_for(mid)
            .map_err(|e| Error::Other(format!("read_var_for({name}): {e}")))?;
        let var_df = record_batch_to_dataframe(&var_batch)?;
        let sce = R!("
            if (!requireNamespace('SingleCellExperiment', quietly = TRUE))
                stop('SingleCellExperiment is required for to_mae()')
            counts_t <- Matrix::t({{dgc}})
            SingleCellExperiment::SingleCellExperiment(
                assays = list(counts = counts_t),
                rowData = S4Vectors::DataFrame({{var_df}})
            )
        ")
        .map_err(|e| Error::Other(format!("SCE for modality '{name}': {e}")))?;
        sce_pairs.push((name.clone(), sce));
    }

    // Assemble experiments(list) on the R side and wrap in a MAE.
    // We feed the SCEs in one at a time via R variable bindings to
    // avoid an arbitrarily-long single R!() call.
    let r_list_init = R!("list()").map_err(|e| Error::Other(format!("init list: {e}")))?;
    let mut exp_list = r_list_init;
    for (name, sce) in &sce_pairs {
        exp_list = R!("
            l <- {{exp_list}}
            l[[{{name.as_str()}}]] <- {{sce}}
            l
        ")
        .map_err(|e| Error::Other(format!("append experiment '{name}': {e}")))?;
    }

    let mae = R!("
        if (!requireNamespace('MultiAssayExperiment', quietly = TRUE))
            stop('MultiAssayExperiment is required for to_mae()')
        MultiAssayExperiment::MultiAssayExperiment(
            experiments = {{exp_list}},
            colData = S4Vectors::DataFrame({{obs_df}})
        )
    ")
    .map_err(|e| Error::Other(format!("MAE construction failed: {}", e)))?;
    Ok(mae)
}

// ─── Module Registration ─────────────────────────────────────────────────────

extendr_module! {
    mod interop;
    fn from_seurat;
    fn from_sce;
    fn from_mae;
}
