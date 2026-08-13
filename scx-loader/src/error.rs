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

    /// Two CSR shards claim the same global obs row.
    ///
    /// The loader's row index is keyed on the global row, so a collision does
    /// not fail — it silently drops whichever shard's row loses. This is that
    /// drop, made loud. Either the file's shard row ranges overlap (a merge /
    /// append / compact defect), or shards of two different modalities were
    /// pooled into one group because no `modality_id` was selected.
    #[error(
        "global obs row {global_row} is claimed by two shards in one shard group \
         (group positions {shard_a} and {shard_b}); CSR shard row ranges must be disjoint"
    )]
    ShardRowOverlap {
        global_row: u64,
        shard_a: usize,
        shard_b: usize,
    },
}

pub type Result<T> = std::result::Result<T, LoaderError>;
