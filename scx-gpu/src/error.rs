use thiserror::Error;

/// Errors produced by SCX GPU operations.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    #[error("GDS unavailable: {0}")]
    GdsUnavailable(String),
    #[error("device {0} not found")]
    DeviceNotFound(usize),
    #[error("invalid shard: {0}")]
    InvalidShard(String),
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },
    #[error("codec error: {0}")]
    CodecError(#[from] scx_codec::CodecError),
    #[error("cuBLAS error: {0}")]
    CuBlasError(String),
    #[error("cuSPARSE error: {0}")]
    CuSparseError(String),
    #[error("cuSOLVER error: {0}")]
    CuSolverError(String),
    #[error("cuRAND error: {0}")]
    CuRandError(String),
    #[error("stream error: {0}")]
    StreamError(String),
    #[error("GPU out of memory: {0}")]
    OutOfMemory(String),
    #[error("module load error: {0}")]
    ModuleLoadError(String),
    #[error("cuVS error: {0}")]
    CuVsError(String),
    #[error("library not found: {0}")]
    LibraryNotFound(String),
    #[error("unsupported device layout: {0}")]
    UnsupportedLayout(String),
}

pub type Result<T> = std::result::Result<T, GpuError>;

impl scx_format_io::PrefetchError for GpuError {
    /// Keeps the pre-4.2 wording of the staging loop's read-error arm, so the
    /// message a user sees when a shard fails to decode mid-stream is unchanged
    /// by the move to the shared decode-prefetch pipeline.
    fn from_shard_read(shard_idx: usize, err: scx_format_io::ScxError) -> Self {
        GpuError::InvalidShard(format!(
            "SCX read error during streaming decode (shard {shard_idx}): {err}"
        ))
    }

    /// A decode worker panicked, or the prefetch channel closed early. Neither
    /// is a CUDA fault, but `InvalidShard` is the arm the staging path already
    /// uses for "the host side could not produce a usable shard".
    fn prefetch_internal(msg: String) -> Self {
        GpuError::InvalidShard(msg)
    }
}
