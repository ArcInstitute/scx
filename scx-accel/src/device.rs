//! The `device=` string grammar shared by the bindings.
//!
//! Every accelerator entry point takes `device="auto"|"cpu"|"gpu"|"gpu:N"`.
//! This module owns the grammar, its error messages, and the resolution
//! against runtime CUDA availability, so pyscx and rscx cannot drift on
//! either the accepted vocabulary or the diagnostics. The companion
//! intent parse — which must *not* collapse `"auto"` — is
//! [`DeviceRequest::from_device_str`](crate::route::DeviceRequest::from_device_str);
//! see its docs for why both exist.
//!
//! Error text is a cross-binding contract: pyscx surfaces
//! [`DeviceError::Invalid`] as `ValueError` and [`DeviceError::Unavailable`]
//! as `RuntimeError`, and its test suite pins the messages.

/// Resolution of a `device=` string to either CPU or a specific GPU index.
///
/// Returned by [`resolve_device`]. Callers extract the GPU index via
/// [`ResolvedDevice::gpu_id`] and forward it into `GpuDevice::new(device_id)`
/// / `*_gpu(device_id, …)` kernel entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedDevice {
    /// Run on the CPU.
    Cpu,
    /// Run on this CUDA device ordinal.
    #[cfg(feature = "gpu")]
    Gpu(usize),
}

impl ResolvedDevice {
    /// Whether this resolution selects a GPU.
    pub fn is_gpu(self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self, ResolvedDevice::Gpu(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// The CUDA device ordinal, or `None` for CPU.
    #[cfg(feature = "gpu")]
    pub fn gpu_id(self) -> Option<usize> {
        match self {
            ResolvedDevice::Gpu(i) => Some(i),
            ResolvedDevice::Cpu => None,
        }
    }
}

/// Why a `device=` string could not be resolved.
///
/// The split mirrors the Python exception taxonomy the messages were written
/// for: `Invalid` is a vocabulary/grammar error (pyscx `ValueError`);
/// `Unavailable` is a well-formed request the runtime cannot satisfy — no
/// CUDA device, out-of-range ordinal, or a build without the `gpu` feature
/// (pyscx `RuntimeError`).
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    /// The string is not in the `auto|cpu|gpu|gpu:N` vocabulary.
    #[error("{0}")]
    Invalid(String),
    /// The request is well-formed but cannot be satisfied at runtime.
    #[error("{0}")]
    Unavailable(String),
}

/// Resolve the device string to a [`ResolvedDevice`].
///
/// Accepted forms:
/// * `"cpu"` → [`ResolvedDevice::Cpu`].
/// * `"auto"` → GPU 0 if the `gpu` feature is enabled and a CUDA device is
///   visible, else [`ResolvedDevice::Cpu`].
/// * `"gpu"` → GPU 0 (errors if no CUDA device is available, or on a build
///   without the `gpu` feature).
/// * `"gpu:N"` for `N: usize` → GPU N (validated against the CUDA device
///   count; out-of-range errors with [`DeviceError::Unavailable`]).
pub fn resolve_device(device: &str) -> Result<ResolvedDevice, DeviceError> {
    if device == "cpu" {
        return Ok(ResolvedDevice::Cpu);
    }
    if device == "auto" {
        #[cfg(feature = "gpu")]
        {
            return Ok(if crate::gpu_available() {
                ResolvedDevice::Gpu(0)
            } else {
                ResolvedDevice::Cpu
            });
        }
        #[cfg(not(feature = "gpu"))]
        {
            return Ok(ResolvedDevice::Cpu);
        }
    }
    if let Some(suffix) = device.strip_prefix("gpu") {
        let device_id = parse_gpu_suffix(device, suffix)?;
        #[cfg(feature = "gpu")]
        {
            if !crate::gpu_available() {
                return Err(DeviceError::Unavailable(format!(
                    "device='{device}' requested but no CUDA GPU found"
                )));
            }
            let count = crate::GpuDevice::count().map_err(|e| {
                DeviceError::Unavailable(format!("failed to query CUDA device count: {e}"))
            })?;
            if device_id >= count {
                return Err(DeviceError::Unavailable(format!(
                    "device='{device}' but only {count} CUDA device{} visible",
                    if count == 1 { " is" } else { "s are" }
                )));
            }
            return Ok(ResolvedDevice::Gpu(device_id));
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = device_id;
            return Err(DeviceError::Unavailable(format!(
                "device='{device}' requested but pyscx was built without the 'gpu' feature"
            )));
        }
    }
    Err(DeviceError::Invalid(format!(
        "unknown device: '{device}'. Use 'auto', 'cpu', 'gpu', or 'gpu:N'."
    )))
}

/// Parse the `:N` suffix of a `"gpu[:N]"` device string. `""` (bare `"gpu"`)
/// resolves to device 0; `":N"` parses `N` as `usize`. Anything else is a
/// [`DeviceError::Invalid`].
fn parse_gpu_suffix(full: &str, suffix: &str) -> Result<usize, DeviceError> {
    if suffix.is_empty() {
        return Ok(0);
    }
    let Some(rest) = suffix.strip_prefix(':') else {
        return Err(DeviceError::Invalid(format!(
            "unknown device: '{full}'. Use 'auto', 'cpu', 'gpu', or 'gpu:N'."
        )));
    };
    rest.parse::<usize>().map_err(|_| {
        DeviceError::Invalid(format!(
            "device='{full}' has a non-integer suffix; expected 'gpu:N' with N a non-negative integer."
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_and_auto_always_resolve() {
        assert_eq!(resolve_device("cpu").unwrap(), ResolvedDevice::Cpu);
        // "auto" never errors, whatever the host: it collapses to CPU when no
        // GPU is reachable.
        assert!(resolve_device("auto").is_ok());
    }

    /// The grammar rejections and their messages are the cross-binding
    /// contract (pyscx surfaces them as `ValueError`, pinned by
    /// `test_accel_gpu_device.py`).
    #[test]
    fn vocabulary_errors_are_invalid_with_the_pinned_text() {
        let err = resolve_device("tpu").unwrap_err();
        assert!(matches!(err, DeviceError::Invalid(_)));
        assert_eq!(
            err.to_string(),
            "unknown device: 'tpu'. Use 'auto', 'cpu', 'gpu', or 'gpu:N'."
        );

        // "gpuX" (no colon) is a vocabulary error, not a suffix parse error.
        let err = resolve_device("gpu0").unwrap_err();
        assert!(matches!(err, DeviceError::Invalid(_)));
        assert_eq!(
            err.to_string(),
            "unknown device: 'gpu0'. Use 'auto', 'cpu', 'gpu', or 'gpu:N'."
        );

        let err = resolve_device("gpu:x").unwrap_err();
        assert!(matches!(err, DeviceError::Invalid(_)));
        assert_eq!(
            err.to_string(),
            "device='gpu:x' has a non-integer suffix; expected 'gpu:N' with N a non-negative integer."
        );
        // A negative ordinal is a non-integer suffix for usize purposes.
        assert!(matches!(
            resolve_device("gpu:-1").unwrap_err(),
            DeviceError::Invalid(_)
        ));
    }

    /// An explicit GPU request on a host/build without one is `Unavailable`
    /// (pyscx `RuntimeError`), never silently CPU. Only assertable on a
    /// non-gpu build here; the gpu-build arm is covered by the pyscx GPU
    /// test suite on a GPU node.
    #[cfg(not(feature = "gpu"))]
    #[test]
    fn explicit_gpu_without_the_feature_is_unavailable() {
        for d in ["gpu", "gpu:1"] {
            let err = resolve_device(d).unwrap_err();
            assert!(matches!(err, DeviceError::Unavailable(_)));
            assert_eq!(
                err.to_string(),
                format!("device='{d}' requested but pyscx was built without the 'gpu' feature")
            );
        }
    }
}
