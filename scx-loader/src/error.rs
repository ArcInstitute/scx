use thiserror::Error;

/// Errors produced by the SCX training data loader.
#[derive(Debug, Error)]
pub enum LoaderError {
    /// Invalid configuration (e.g., batch_size=0).
    #[error("loader configuration error: {reason}")]
    ConfigError { reason: String },

    /// File I/O failure.
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Propagated from scx-format.
    #[error(transparent)]
    FormatError(#[from] scx_format_io::error::ScxError),

    /// Propagated from Arrow.
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),

    /// Bounded channel send/receive failure.
    #[error("channel error: {0}")]
    ChannelError(String),

    /// Pipeline stage panicked or exited unexpectedly.
    #[error("pipeline shutdown error: {0}")]
    ShutdownError(String),

    /// Plan-driven row index referenced a row not present in the file.
    #[error("row index {idx} is out of range (n_obs={n_obs})")]
    IndexOutOfRange { idx: u64, n_obs: usize },

    /// Requested obs column is not present in the file's metadata.
    #[error("obs column '{name}' not found in file (available: {available:?})")]
    ObsColumnNotFound {
        name: String,
        available: Vec<String>,
    },
}

pub type Result<T> = std::result::Result<T, LoaderError>;
