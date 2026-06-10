//! Error types for MTX I/O.

use scx_format::error::ScxError;

#[derive(Debug, thiserror::Error)]
pub enum MtxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SCX error: {0}")]
    Scx(#[from] ScxError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("MTX parse error: {0}")]
    Parse(String),

    #[error("missing required file: {0}")]
    MissingFile(String),

    #[error(
        "MTX orientation mismatch: matrix is {n_rows}×{n_cols}, but barcodes.tsv has \
         {n_barcodes} entries and features.tsv has {n_features}; neither \
         (features×barcodes = {n_features}×{n_barcodes}) nor \
         (barcodes×features = {n_barcodes}×{n_features}) matches the matrix dimensions"
    )]
    OrientationMismatch {
        n_rows: usize,
        n_cols: usize,
        n_barcodes: usize,
        n_features: usize,
    },

    #[error("invalid codec: {0}")]
    InvalidCodec(String),

    #[error("{0}")]
    Other(String),
}
