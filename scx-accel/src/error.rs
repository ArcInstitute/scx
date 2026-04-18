use scx_engine::EngineError;
use scx_format::ScxError;

/// Errors from accelerator operations.
#[derive(Debug, thiserror::Error)]
pub enum AccelError {
    #[error("SCX format error: {0}")]
    Scx(#[from] ScxError),

    #[error("SCX engine error: {0}")]
    Engine(#[from] EngineError),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("linear algebra error: {0}")]
    LinAlg(String),

    #[error("statistics error: {0}")]
    StatsError(String),

    #[error("shape error: {0}")]
    ShapeError(String),

    #[error("numerical instability: {0}")]
    NumericalInstability(String),
}

pub type Result<T> = std::result::Result<T, AccelError>;
