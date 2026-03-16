use thiserror::Error;

/// Errors produced by the SCX format crate.
#[derive(Debug, Error)]
pub enum ScxError {
    #[error("invalid file magic bytes")]
    InvalidMagic,

    #[error("unsupported format version")]
    UnsupportedVersion,

    #[error("unsupported endianness (only little-endian is supported)")]
    UnsupportedEndian,

    #[error("checksum mismatch")]
    ChecksumMismatch,

    #[error("invalid shard magic bytes")]
    InvalidShardMagic,

    #[error("unknown codec ID: {0}")]
    UnknownCodec(u8),

    #[error("unknown value encoding: {0}")]
    UnknownValueEncoding(u8),

    #[error("inconsistent CSR array lengths")]
    InconsistentCsr,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
}

pub type Result<T> = std::result::Result<T, ScxError>;
