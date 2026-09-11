// Read 10x Genomics HDF5 files

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

use super::pipeline::ConvertError;
use crate::h5ad::read::{read_f32_dataset, read_i32_dataset, read_i64_dataset};
use crate::h5ad::strings::read_string_array;

pub struct TenXData {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
    pub n_cells: usize,
    pub n_genes: usize,
    pub obs: RecordBatch,
    pub var: RecordBatch,
}

/// Read `/matrix/shape` as `(n_genes, n_cells)` — **10x axis order**, which is
/// the transpose of the `(n_obs, n_vars)` that [`crate::h5ad::read::read_shape_2d`]
/// returns for an h5ad sparse group. The axis order is in the name because a
/// caller that swaps it gets an obs axis of genes and a matrix that only fails
/// much later.
///
/// Real CellRanger (and CellBender) files write `/matrix/shape` as a *dataset*;
/// some synthetic fixtures write it as an attribute. Accept either — reading
/// only the attribute fails on real 10x output, and reading only the dataset
/// fails on the fixtures.
pub(crate) fn read_tenx_shape(matrix: &hdf5::Group) -> Result<(usize, usize), ConvertError> {
    let shape: Vec<i64> = match matrix.dataset("shape") {
        Ok(ds) => read_i64_dataset(&ds)?,
        Err(_) => matrix
            .attr("shape")
            .map_err(|_| {
                ConvertError::Other(
                    "10x file has neither a `/matrix/shape` dataset nor a `shape` attribute"
                        .to_string(),
                )
            })?
            .read_1d()?
            .to_vec(),
    };
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "/matrix/shape must have 2 entries, found {}",
            shape.len()
        )));
    }
    let n_genes = shape[0] as usize;
    let n_cells = shape[1] as usize;
    Ok((n_genes, n_cells))
}

/// Build `obs` from `/matrix/barcodes`: one `barcode` column, no nulls.
pub(crate) fn read_tenx_obs(matrix: &hdf5::Group) -> Result<RecordBatch, ConvertError> {
    let barcodes_ds = matrix.dataset("barcodes")?;
    let obs_schema = Schema::new(vec![Field::new("barcode", DataType::Utf8, false)]);
    Ok(RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![Arc::new(read_string_array(&barcodes_ds)?) as ArrayRef],
    )?)
}

/// Build `var` from `/matrix/features`, keeping each column under its source
/// name. Every column is optional; the group is not. When none of the three is
/// present, fabricate `gene_id` so the var axis still has an identity.
pub(crate) fn read_tenx_var(
    matrix: &hdf5::Group,
    n_genes: usize,
) -> Result<RecordBatch, ConvertError> {
    let features = matrix.group("features")?;
    let mut var_fields = Vec::new();
    let mut var_arrays: Vec<ArrayRef> = Vec::new();

    for col_name in &["id", "name", "feature_type"] {
        if let Ok(ds) = features.dataset(col_name) {
            var_fields.push(Field::new(*col_name, DataType::Utf8, false));
            var_arrays.push(Arc::new(read_string_array(&ds)?));
        }
    }

    Ok(if var_fields.is_empty() {
        // Fallback: create dummy var
        let ids: Vec<String> = (0..n_genes).map(|i| format!("gene_{i}")).collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "gene_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(ids)) as ArrayRef],
        )?
    } else {
        RecordBatch::try_new(Arc::new(Schema::new(var_fields)), var_arrays)?
    })
}

/// Read a 10x Genomics HDF5 file **whole**.
///
/// 10x files store the matrix in CSC format under `matrix/` with
/// shape `[n_genes, n_cells]`. A CSC of `[n_genes × n_cells]` *is* a CSR of
/// `[n_cells × n_genes]` byte for byte, so nothing is transposed here — the
/// `indptr` (length `n_cells + 1`) and the gene `indices` are used verbatim as
/// CSR-over-cells.
///
/// This is the eager path's reader: the three sparse arrays land in memory in
/// full. The streaming path reads the same group one row range at a time
/// through [`crate::h5ad::stream::open_tenx_x_streaming`] and shares the shape
/// / obs / var parses above.
pub fn read_tenx_h5(file: &hdf5::File) -> Result<TenXData, ConvertError> {
    let matrix = file.group("matrix")?;

    let (n_genes, n_cells) = read_tenx_shape(&matrix)?;

    // Read CSC arrays
    // CSC of [n_genes × n_cells] = CSR of [n_cells × n_genes] (transposed)
    // So we can directly reinterpret CSC as CSR of the transposed matrix.
    let csr_indptr = read_i64_dataset(&matrix.dataset("indptr")?)?;
    let csr_indices = read_i32_dataset(&matrix.dataset("indices")?)?;
    let csr_data = read_f32_dataset(&matrix.dataset("data")?)?;

    let obs = read_tenx_obs(&matrix)?;
    let var = read_tenx_var(&matrix, n_genes)?;

    Ok(TenXData {
        indptr: csr_indptr,
        indices: csr_indices,
        data: csr_data,
        n_cells,
        n_genes,
        obs,
        var,
    })
}
