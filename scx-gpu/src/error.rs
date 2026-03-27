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
}

pub type Result<T> = std::result::Result<T, GpuError>;
