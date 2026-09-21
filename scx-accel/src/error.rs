use scx_engine::EngineError;
use scx_format_io::ScxError;

/// A user-facing argument-vocabulary error whose `Display` is exactly the
/// canonical cross-binding message, with no category prefix.
///
/// Returned by the shared `parse` constructors (`ScoreMethod::parse`,
/// `AggregationMethod::parse`, `DispersionMethod::parse`) so each binding can
/// surface the text verbatim — pyscx as `ValueError`/`RuntimeError`, rscx as
/// an R error — without stripping the `"invalid input: "` prefix
/// [`AccelError::InvalidInput`] would add. The message text is a contract:
/// the pyscx test suite pins it.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct InvalidArgument(pub String);

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

    /// The GPU does not have enough free memory for the op, and no non-GPU
    /// path was substituted (`device="auto"` resolves the device up front and
    /// does not re-run on CPU after a runtime failure — see
    /// [`crate::route::DeviceRequest::Auto`]). The inner string must name the
    /// shortfall in the terms the user controls — how much the op needs, how
    /// much was free, and what to change — because that message is the whole
    /// remedy the caller gets. Surfaced as `RuntimeError` on the Python side.
    #[error("GPU out of memory: {0}")]
    GpuOutOfMemory(String),

    /// `prefer_format="csc"` was requested but the dataset cannot
    /// service CSC reads. The inner string names the missing
    /// capability — for example: no CSC sidecar on disk, or an active
    /// row deletion vector (which renumbers the live rows while CSC
    /// `indices` stay global). Surfaced as `RuntimeError` on the Python
    /// side.
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
