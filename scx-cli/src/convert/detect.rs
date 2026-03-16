// Format detection for h5ad vs 10x files

use hdf5::types::VarLenUnicode;

use super::pipeline::ConvertError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    H5ad,
    TenX,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixFormat {
    Csr,
    Csc,
    Dense,
}

/// Detect whether an HDF5 file is h5ad or 10x format.
pub fn detect_input_format(file: &hdf5::File) -> Result<InputFormat, ConvertError> {
    // h5ad files have an "obs" group at the root
    if file.group("obs").is_ok() {
        return Ok(InputFormat::H5ad);
    }
    // 10x files have a "matrix" group with "barcodes"
    if let Ok(matrix) = file.group("matrix") {
        if matrix.dataset("barcodes").is_ok() {
            return Ok(InputFormat::TenX);
        }
    }
    Err(ConvertError::Other(
        "cannot detect input format: file has neither 'obs' group (h5ad) nor 'matrix/barcodes' (10x)"
            .to_string(),
    ))
}

/// Detect the matrix storage format in an h5ad file.
pub fn detect_matrix_format(file: &hdf5::File) -> Result<MatrixFormat, ConvertError> {
    // Check if X is a group (sparse) or dataset (dense)
    if let Ok(group) = file.group("X") {
        // Check encoding-type attribute
        if let Ok(attr) = group.attr("encoding-type") {
            let encoding_type: VarLenUnicode = attr.read_scalar()?;
            let encoding_str = encoding_type.as_str();
            return match encoding_str {
                "csr_matrix" => Ok(MatrixFormat::Csr),
                "csc_matrix" => Ok(MatrixFormat::Csc),
                "array" => Ok(MatrixFormat::Dense),
                other => Err(ConvertError::UnsupportedDtype(format!(
                    "unknown encoding-type: {other}"
                ))),
            };
        }
        // Fallback: check for sparse structure
        if group.dataset("indptr").is_ok() && group.dataset("indices").is_ok() {
            // Default to CSR per h5ad convention
            return Ok(MatrixFormat::Csr);
        }
        // Group with data but no sparse structure
        if group.dataset("data").is_ok() {
            return Ok(MatrixFormat::Csr);
        }
    }
    // X is a dataset → dense
    if file.dataset("X").is_ok() {
        return Ok(MatrixFormat::Dense);
    }
    Err(ConvertError::Other(
        "cannot find X matrix in h5ad file".to_string(),
    ))
}
