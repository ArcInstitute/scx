use thiserror::Error;

/// Errors produced by SCX cloud operations.
#[derive(Debug, Error)]
pub enum CloudError {
    #[error("SCX format error: {0}")]
    Format(#[from] scx_format::ScxError),

    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    #[error("Invalid cloud URL: {0}")]
    InvalidUrl(String),

    #[error("Catalog not found at {0} — is this a valid .scxd directory?")]
    CatalogNotFound(String),

    #[error("Section not found: {0}")]
    SectionNotFound(String),

    #[error("Authentication failed for {backend}: {reason}")]
    AuthError { backend: String, reason: String },

    #[error("Download failed after {retries} retries: {message}")]
    DownloadFailed { retries: usize, message: String },

    #[error("Pull interrupted: {0}")]
    Interrupted(String),

    #[error("Engine error: {0}")]
    Engine(#[from] scx_engine::EngineError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error(
        "slice bounds out of range: offset {offset} + length {length} exceeds data size {data_len}"
    )]
    SliceBoundsExceeded {
        offset: usize,
        length: usize,
        data_len: usize,
    },
}

pub type Result<T> = std::result::Result<T, CloudError>;
