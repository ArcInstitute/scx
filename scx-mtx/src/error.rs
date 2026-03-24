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

    #[error("{0}")]
    Other(String),
}
