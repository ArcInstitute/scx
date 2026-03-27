use scx_format::ScxError;

/// Errors from accelerator operations.
#[derive(Debug, thiserror::Error)]
pub enum AccelError {
    #[error("SCX format error: {0}")]
    Scx(#[from] ScxError),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("linear algebra error: {0}")]
    LinAlg(String),
}

pub type Result<T> = std::result::Result<T, AccelError>;
