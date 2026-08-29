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
    /// A device allocation, host synchronization or module load was attempted
    /// inside a CUDA-graph capture region. Raised by
    /// [`crate::capture_guard::check`]; see that module for why the contract is
    /// enforced by a returned error rather than a `debug_assert!`.
    #[error("CUDA-graph capture contract violated: {0}")]
    CaptureViolation(String),
}

pub type Result<T> = std::result::Result<T, GpuError>;

impl GpuError {
    /// Whether the *device* failed, as opposed to the input being defective.
    ///
    /// True means: this same input, handed to a non-GPU path, would have
    /// produced an answer — the GPU ran out of memory, a driver/library call
    /// failed, a module would not load. Those are the failures a caller may
    /// legitimately answer by taking another path.
    ///
    /// False means the input itself is the problem, so no other path helps:
    ///
    /// * [`GpuError::InvalidShard`] — the shard is malformed, and the CPU
    ///   kernels reject exactly the same shards (`shard_validate::validate_shard`
    ///   exists precisely because a NaN or a
    ///   duplicate `(row, col)` is not computable anywhere). It also carries
    ///   the *host-side* read/decode failures via
    ///   [`scx_format_io::PrefetchError`] below, which no device would fix
    ///   either. Retrying these elsewhere converts a fast, precise rejection
    ///   into a slow one.
    /// * [`GpuError::ShapeMismatch`] / [`GpuError::CodecError`] — a caller bug
    ///   or an unreadable stream; deterministic in both cases.
    /// * [`GpuError::UnsupportedLayout`] — a routing decision, already
    ///   expressible as `FallbackReason::UnsupportedInputLayout`; it is not a
    ///   failure that happened, it is a path that was never viable.
    ///
    /// The `match` is deliberately exhaustive with no `_` arm: a new
    /// `GpuError` variant must be classified by whoever adds it, and the
    /// compiler — not a test — is what enforces that.
    pub fn is_runtime_failure(&self) -> bool {
        match self {
            GpuError::CudaError(_)
            | GpuError::KernelLaunchFailed(_)
            | GpuError::GdsUnavailable(_)
            | GpuError::DeviceNotFound(_)
            | GpuError::CuBlasError(_)
            | GpuError::CuSparseError(_)
            | GpuError::CuSolverError(_)
            | GpuError::CuRandError(_)
            | GpuError::StreamError(_)
            | GpuError::OutOfMemory(_)
            | GpuError::ModuleLoadError(_)
            | GpuError::CuVsError(_)
            | GpuError::LibraryNotFound(_)
            // A capture violation is a defect in *our* code, not in the input —
            // and by the rule above that puts it here: the same input run
            // without graph capture produces the answer, which is exactly the
            // fallback `harmony/gpu.rs` already takes when a capture fails.
            | GpuError::CaptureViolation(_) => true,

            GpuError::InvalidShard(_)
            | GpuError::ShapeMismatch { .. }
            | GpuError::CodecError(_)
            | GpuError::UnsupportedLayout(_) => false,
        }
    }

    /// Whether producing the same result by a **different route to the same
    /// device** could succeed.
    ///
    /// [`Self::is_runtime_failure`] minus out-of-memory. A module that will not
    /// load, a kernel launch failure, a library problem — another route (decode
    /// on the host, then upload) does not go near any of those. A memory
    /// shortfall is different in kind: every route still has to land the same
    /// bytes in the same VRAM, so an alternative can only buy a slower trip to
    /// the same error, and with a worse message than the one already in hand.
    ///
    /// Used by `Experiment.to_gpu_anndata` to decide whether a failed in-VRAM
    /// shard decode falls through to host-assemble. Note this is *not* the
    /// question a GPU→CPU fallback would ask — a CPU kernel needs no VRAM at
    /// all, so out-of-memory would belong on its "yes" side. There is no such
    /// fallback: `device="auto"` resolves the device before the op starts and
    /// raises on a runtime failure rather than re-running on CPU.
    pub fn alternate_route_may_succeed(&self) -> bool {
        self.is_runtime_failure() && !matches!(self, GpuError::OutOfMemory(_))
    }
}

/// Turn a device-side failure while building an *optional* accelerator into a
/// decline, leaving the caller's slower-but-equivalent path to run.
///
/// For builders that already return `Ok(None)` to mean "declined, stream
/// instead" — [`crate::try_build_resident`] and the PCA resident-CSR builder.
/// Both are optimisations layered over a streaming path that produces the same
/// answer, so a device failure while *constructing* one must not be the reason
/// the whole op errors. An input-defect error ([`GpuError::is_runtime_failure`]
/// = false) still propagates: the streaming path would only re-derive it.
///
/// Callers are responsible for leaving the source re-drivable after a partial
/// drain; both current callers do (`RawGpuShardSource::run` drains its pinned
/// events on every exit path, and `try_build_resident` returns its retained
/// device buffers to the pool before propagating).
pub fn decline_on_runtime_failure<T>(built: Result<Option<T>>, what: &str) -> Result<Option<T>> {
    match built {
        Ok(v) => Ok(v),
        Err(e) if e.is_runtime_failure() => {
            log::warn!("{what} declined ({e}); falling back to the streaming path");
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn s() -> String {
        "boom".to_string()
    }

    /// Every variant, both classes. Listing them one by one (rather than
    /// looping) is the point: adding a variant breaks the exhaustive `match`
    /// in `is_runtime_failure` at compile time, and this test is where the
    /// intended answer for each existing variant is written down.
    #[test]
    fn device_failures_are_runtime_failures() {
        for e in [
            GpuError::CudaError(s()),
            GpuError::KernelLaunchFailed(s()),
            GpuError::GdsUnavailable(s()),
            GpuError::DeviceNotFound(3),
            GpuError::CuBlasError(s()),
            GpuError::CuSparseError(s()),
            GpuError::CuSolverError(s()),
            GpuError::CuRandError(s()),
            GpuError::StreamError(s()),
            GpuError::OutOfMemory(s()),
            GpuError::ModuleLoadError(s()),
            GpuError::CuVsError(s()),
            GpuError::LibraryNotFound(s()),
            GpuError::CaptureViolation(s()),
        ] {
            assert!(e.is_runtime_failure(), "{e} should be a runtime failure");
        }
    }

    #[test]
    fn input_defects_are_not_runtime_failures() {
        for e in [
            GpuError::InvalidShard(s()),
            GpuError::ShapeMismatch {
                expected: s(),
                got: s(),
            },
            GpuError::CodecError(scx_codec::CodecError::MalformedInput(s())),
            GpuError::UnsupportedLayout(s()),
        ] {
            assert!(
                !e.is_runtime_failure(),
                "{e} is an input defect, not a device failure"
            );
        }
    }

    /// A host-side shard read failure arrives as `InvalidShard` through the
    /// `PrefetchError` impl above. It must stay on the non-runtime side: no
    /// device can fix an unreadable file, so declining to the streaming path
    /// would only re-derive the same error one pass later.
    #[test]
    fn a_host_side_read_failure_is_not_a_runtime_failure() {
        use scx_format_io::PrefetchError;
        let e = GpuError::prefetch_internal("decode worker panicked".to_string());
        assert!(!e.is_runtime_failure());
    }

    /// The out-of-memory carve-out is a design decision, not an oversight, and
    /// nothing else records it: both routes to the device end with the same CSR
    /// resident in the same VRAM, so an alternative route cannot fix a
    /// shortfall — it can only pay a full host materialization to fail again.
    #[test]
    fn out_of_memory_is_the_one_device_failure_no_alternate_route_survives() {
        let oom = GpuError::OutOfMemory(s());
        assert!(oom.is_runtime_failure());
        assert!(!oom.alternate_route_may_succeed());
    }

    #[test]
    fn other_device_failures_admit_an_alternate_route() {
        for e in [
            // The realistic trigger: a build whose PTX did not compile bakes
            // empty stubs that fail here, and the host decode is untouched.
            GpuError::ModuleLoadError(s()),
            GpuError::KernelLaunchFailed(s()),
            GpuError::CudaError(s()),
            GpuError::LibraryNotFound(s()),
        ] {
            assert!(e.alternate_route_may_succeed(), "{e}");
        }
    }

    #[test]
    fn an_input_defect_admits_no_alternate_route() {
        assert!(!GpuError::InvalidShard(s()).alternate_route_may_succeed());
        assert!(!GpuError::UnsupportedLayout(s()).alternate_route_may_succeed());
    }

    #[test]
    fn decline_passes_success_through_unchanged() {
        let built: Result<Option<u8>> = Ok(Some(7));
        assert_eq!(
            decline_on_runtime_failure(built, "widget").unwrap(),
            Some(7)
        );

        let declined: Result<Option<u8>> = Ok(None);
        assert_eq!(
            decline_on_runtime_failure(declined, "widget").unwrap(),
            None
        );
    }

    #[test]
    fn decline_converts_a_device_failure_into_a_decline() {
        let built: Result<Option<()>> = Err(GpuError::OutOfMemory("clone_exact".to_string()));
        assert_eq!(
            decline_on_runtime_failure(built, "resident CSR").unwrap(),
            None
        );
    }

    #[test]
    fn decline_still_propagates_an_input_defect() {
        let built: Result<Option<()>> = Err(GpuError::InvalidShard("NaN at 3".to_string()));
        let err = decline_on_runtime_failure(built, "resident CSR")
            .expect_err("a malformed shard must not be swallowed as a decline");
        assert!(matches!(err, GpuError::InvalidShard(_)));
    }
}
