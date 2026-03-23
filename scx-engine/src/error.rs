use thiserror::Error;

/// Errors produced by the SCX query engine.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("schema error on column '{column}': {reason}")]
    SchemaError { column: String, reason: String },

    #[error("predicate parse error in '{expr}': {reason}")]
    PredicateParseError { expr: String, reason: String },

    #[error("collect() called on empty pipeline (no file opened)")]
    EmptyPipeline,

    #[error(transparent)]
    FormatError(#[from] scx_format::ScxError),

    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    CsrError(#[from] scx_sparse::CsrError),
}

pub type Result<T> = std::result::Result<T, EngineError>;
