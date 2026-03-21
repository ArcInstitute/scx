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
    FormatError(#[from] scx_format::error::ScxError),

    /// Propagated from Arrow.
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),

    /// Bounded channel send/receive failure.
    #[error("channel error: {0}")]
    ChannelError(String),

    /// `next_batch()` called before `start_epoch()`.
    #[error("next_batch() called before start_epoch()")]
    EpochNotStarted,

    /// Pipeline stage panicked or exited unexpectedly.
    #[error("pipeline shutdown error: {0}")]
    ShutdownError(String),
}

pub type Result<T> = std::result::Result<T, LoaderError>;
