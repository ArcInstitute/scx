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

    #[error("root catalog exceeds 4096 byte limit: {0} bytes")]
    RootCatalogTooLarge(usize),

    #[error("unknown section type: {0}")]
    UnknownSectionType(u8),

    #[error("section not found: {0}")]
    SectionNotFound(String),

    #[error("shard index {index} out of bounds (count: {count})")]
    ShardIndexOutOfBounds { index: usize, count: usize },

    #[error("CSR error: {0}")]
    Csr(#[from] scx_sparse::CsrError),

    #[error("codec error: {0}")]
    Codec(#[from] scx_codec::CodecError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
}

pub type Result<T> = std::result::Result<T, ScxError>;
