// Read 10x Genomics HDF5 files

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

use super::pipeline::ConvertError;
use crate::h5ad::read::{read_f32_dataset, read_i32_dataset, read_i64_dataset};

pub struct TenXData {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
    pub n_cells: usize,
    pub n_genes: usize,
    pub obs: RecordBatch,
    pub var: RecordBatch,
}

/// Read a 10x Genomics HDF5 file.
///
/// 10x files store the matrix in CSC format under `matrix/` with
/// shape `[n_genes, n_cells]`. This function reads and transposes to CSR.
pub fn read_tenx_h5(file: &hdf5::File) -> Result<TenXData, ConvertError> {
    let matrix = file.group("matrix")?;

    // Read shape: [n_genes, n_cells]
    let shape: Vec<i64> = matrix.attr("shape")?.read_1d()?.to_vec();
    let n_genes = shape[0] as usize;
    let n_cells = shape[1] as usize;

    // Read CSC arrays
    // CSC of [n_genes × n_cells] = CSR of [n_cells × n_genes] (transposed)
    // So we can directly reinterpret CSC as CSR of the transposed matrix.
    let csr_indptr = read_i64_dataset(&matrix.dataset("indptr")?)?;
    let csr_indices = read_i32_dataset(&matrix.dataset("indices")?)?;
    let csr_data = read_f32_dataset(&matrix.dataset("data")?)?;

    // Build obs from barcodes
    let barcodes_ds = matrix.dataset("barcodes")?;
    let barcodes: Vec<hdf5::types::VarLenUnicode> = barcodes_ds.read_1d()?.to_vec();
    let barcode_strings: Vec<String> = barcodes.iter().map(|s| s.to_string()).collect();
    let obs_schema = Schema::new(vec![Field::new("barcode", DataType::Utf8, false)]);
    let obs = RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![Arc::new(StringArray::from(
            barcode_strings
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )) as ArrayRef],
    )?;

    // Build var from features
    let features = matrix.group("features")?;
    let mut var_fields = Vec::new();
    let mut var_arrays: Vec<ArrayRef> = Vec::new();

    for col_name in &["id", "name", "feature_type"] {
        if let Ok(ds) = features.dataset(col_name) {
            let data: Vec<hdf5::types::VarLenUnicode> = ds.read_1d()?.to_vec();
            let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
            var_fields.push(Field::new(*col_name, DataType::Utf8, false));
            var_arrays.push(Arc::new(StringArray::from(
                strings.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )));
        }
    }

    let var = if var_fields.is_empty() {
        // Fallback: create dummy var
        let ids: Vec<String> = (0..n_genes).map(|i| format!("gene_{i}")).collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "gene_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef],
        )?
    } else {
        RecordBatch::try_new(Arc::new(Schema::new(var_fields)), var_arrays)?
    };

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
