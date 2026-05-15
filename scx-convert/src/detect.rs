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

/// Detect the matrix storage format in an h5ad file (X at root).
pub fn detect_matrix_format(file: &hdf5::File) -> Result<MatrixFormat, ConvertError> {
    detect_matrix_format_at(file, "X")
}

/// Detect matrix storage format for a matrix at an arbitrary path inside
/// an HDF5 file (e.g. `mod/rna/X` for a per-modality matrix in an h5mu
/// file). Returns the same `MatrixFormat` semantics as the X-at-root
/// detector — Csr / Csc for sparse groups, Dense for plain datasets.
pub fn detect_matrix_format_at(
    file: &hdf5::File,
    path: &str,
) -> Result<MatrixFormat, ConvertError> {
    // Check if path is a group (sparse) or dataset (dense)
    if let Ok(group) = file.group(path) {
        // Check encoding-type attribute
        if let Ok(attr) = group.attr("encoding-type") {
            let encoding_type: VarLenUnicode = attr.read_scalar()?;
            let encoding_str = encoding_type.as_str();
            return match encoding_str {
                "csr_matrix" => Ok(MatrixFormat::Csr),
                "csc_matrix" => Ok(MatrixFormat::Csc),
                "array" => Ok(MatrixFormat::Dense),
                other => Err(ConvertError::UnsupportedDtype(format!(
                    "unknown encoding-type at '{path}': {other}"
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
    // path is a dataset → dense
    if file.dataset(path).is_ok() {
        return Ok(MatrixFormat::Dense);
    }
    Err(ConvertError::Other(format!(
        "cannot find matrix at '{path}'"
    )))
}
