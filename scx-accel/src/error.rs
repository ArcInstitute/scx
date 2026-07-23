use scx_engine::EngineError;
use scx_format_io::ScxError;

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

    /// GPU device initialization failed (no driver, context creation error,
    /// out of memory, …) after a GPU route was requested. An explicit
    /// `device="gpu"` must surface this to the caller rather than silently
    /// running the CPU kernel under a GPU route stamp (§4.1). Surfaced as
    /// `RuntimeError` on the Python side.
    #[error("GPU device initialization failed: {0}")]
    GpuInitFailed(String),

    /// `prefer_format="csc"` was requested but the dataset cannot
    /// service CSC reads. The inner string names the missing
    /// capability — for example: no CSC sidecar on disk, a
    /// non-column-local transform in the chain, or an active row
    /// deletion vector. Surfaced as `RuntimeError` on the Python side.
    #[error("CSC requested but unavailable: {0}")]
    CscRequestedNotAvailable(String),

    /// Sentinel returned by `csc::require_csc()` when the caller passed
    /// `PreferFormat::Csr`. Lets callers branch on the return type
    /// instead of duplicating the kwarg check at each call site;
    /// never escapes to Python (callers map this to the CSR path).
    #[error("CSC not requested (caller chose CSR)")]
    CscNotRequested,
}

pub type Result<T> = std::result::Result<T, AccelError>;
