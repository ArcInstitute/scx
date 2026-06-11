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

    #[error("unknown index preset '{0}'; expected one of cellxgene, perturbseq, training")]
    UnknownIndexPreset(String),

    #[error(transparent)]
    FormatError(#[from] scx_format_io::ScxError),

    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    CsrError(#[from] scx_sparse::CsrError),

    /// Free-form error for builder/contract violations that don't
    /// warrant a dedicated variant. Used by
    /// [`crate::ObsPredicateIndexBuilder`] for shard-order /
    /// schema-shape checks where the caller is the offender.
    #[error("{0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
