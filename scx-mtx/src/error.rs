//! Error types for MTX I/O.

use scx_format_io::error::ScxError;

#[derive(Debug, thiserror::Error)]
pub enum MtxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SCX error: {0}")]
    Scx(#[from] ScxError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("MTX parse error: {0}")]
    Parse(String),

    #[error("missing required file: {0}")]
    MissingFile(String),

    #[error(
        "MTX orientation mismatch: matrix is {n_rows}×{n_cols}, but barcodes.tsv has \
         {n_barcodes} entries and features.tsv has {n_features}; neither \
         (features×barcodes = {n_features}×{n_barcodes}) nor \
         (barcodes×features = {n_barcodes}×{n_features}) matches the matrix dimensions"
    )]
    OrientationMismatch {
        n_rows: usize,
        n_cols: usize,
        n_barcodes: usize,
        n_features: usize,
    },

    #[error(
        "MTX export requires a single modality: this file has {} \
         ({}). Pass --modality NAME (pyscx: modality=\"NAME\") to choose one — a \
         MatrixMarket directory describes one matrix over one feature space, and \
         exporting a multimodal file unscoped stacks the modalities into one \
         matrix over mixed column spaces.",
        available.len(),
        available.join(", ")
    )]
    ModalityRequired { available: Vec<String> },

    #[error("unknown modality '{requested}'; this file has: {}", available.join(", "))]
    UnknownModality {
        requested: String,
        available: Vec<String>,
    },

    #[error("input file is single-modality; `--modality {requested}` is not applicable")]
    ModalityNotApplicable { requested: String },

    #[error(
        "integer value {value} exceeds 2\u{b2}\u{2074} ({}), the largest integer representable \
         exactly in float32; MTX ingest carries values as f32, so storing it would \
         silently round the count. Pass --allow-lossy (pyscx: allow_lossy=True) to \
         accept the rounding.",
        scx_codec::F32_MAX_EXACT_INT
    )]
    LossyIntegerValue { value: i64 },

    #[error(
        "duplicate coordinates at ({row}, {col}) sum past 2\u{b2}\u{2074} ({}), the largest \
         integer representable exactly in float32; MatrixMarket sums duplicates, so \
         the stored count would be rounded. Pass --allow-lossy (pyscx: \
         allow_lossy=True) to accept the rounding.",
        scx_codec::F32_MAX_EXACT_INT
    )]
    LossySummedIntegerValue { row: usize, col: usize },

    #[error("invalid codec: {0}")]
    InvalidCodec(String),

    #[error("shard_target_rows must be > 0 (0 would stall the shard-writing loop)")]
    InvalidShardSize,

    #[error("{0}")]
    Other(String),
}
