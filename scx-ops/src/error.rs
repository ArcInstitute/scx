use thiserror::Error;

/// Errors produced by scx-ops operations.
#[derive(Debug, Error)]
pub enum OpsError {
    #[error("format error: {0}")]
    Format(#[from] scx_format::ScxError),

    #[error("failed to acquire file lock: {0}")]
    LockFailed(std::io::Error),

    #[error("manifest sequence mismatch: expected {expected}, found {found}")]
    ManifestMismatch { expected: u64, found: u64 },

    #[error("rollback target sequence {0} not found in catalog chain")]
    RollbackTargetNotFound(u64),

    #[error("incompatible n_vars: expected {expected}, found {found}")]
    IncompatibleVars { expected: u64, found: u64 },

    #[error("no previous catalog available for rollback")]
    NoPreviousCatalog,

    #[error("unknown codec ID: {0}")]
    UnknownCodec(u8),

    #[error("unknown value encoding: {0}")]
    UnknownValueEncoding(u8),

    #[error("CSR index {index} out of bounds for n_vars={n_vars}")]
    IndexOutOfBounds { index: u32, n_vars: u64 },

    #[error("layer '{name}' missing in input file {file_index}")]
    LayerMissing { name: String, file_index: usize },

    #[error("obs schema mismatch on append: {detail}")]
    SchemaMismatch { detail: String },

    #[error(
        "obs batch length mismatch on append: target expects {expected} new rows, \
         new_obs has {found}"
    )]
    VarLengthMismatch { expected: usize, found: usize },

    #[error("CSR shape mismatch on append: {detail}")]
    ShapeMismatch { detail: String },

    #[error("f32 value {value} out of range for {encoding} encoding (max {max})")]
    ValueOutOfRange {
        value: f32,
        encoding: &'static str,
        max: f32,
    },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("codec error: {0}")]
    Codec(#[from] scx_codec::CodecError),

    #[error("modality mismatch on merge: {detail}")]
    ModalityMismatch { detail: String },

    #[error(
        "{op} is not yet supported for multimodal files; \
         extract individual modalities first with \
         `scx subset --modality NAME`"
    )]
    MultimodalUnsupported { op: &'static str },

    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
}

pub type Result<T> = std::result::Result<T, OpsError>;
