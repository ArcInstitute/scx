//! GPU device management wrapping cudarc 0.19 `CudaContext` + `CudaStream`.
//!
//! `GpuDevice` provides a thin ergonomic wrapper over cudarc's low-level
//! CUDA driver API, handling context creation, stream management, memory
//! operations, and module loading. All CUDA errors are mapped to [`GpuError`].

use std::sync::Arc;

use cudarc::driver::safe::{
    CudaContext, CudaModule, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use crate::error::GpuError;

/// A GPU device handle wrapping a CUDA context and its default stream.
///
/// In cudarc 0.19, `CudaDevice` was removed. The primary abstractions are now
/// [`CudaContext`] (device handle) and [`CudaStream`] (work scheduling).
/// Memory operations (`alloc_zeros`, `clone_htod`, `clone_dtoh`) are methods
/// on `CudaStream` via `&Arc<Self>`, not the context.
pub struct GpuDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl GpuDevice {
    /// Create a new `GpuDevice` for the given device ordinal.
    ///
    /// Initializes the CUDA driver (if not already initialized), creates a
    /// context on the specified device, and obtains the default stream.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::DeviceNotFound`] if the ordinal exceeds the number
    /// of available devices, or [`GpuError::CudaError`] on other CUDA failures.
    pub fn new(device_id: usize) -> Result<Self, GpuError> {
        // Ensure CUDA driver is initialized.
        cudarc::driver::result::init()
            .map_err(|e| GpuError::CudaError(format!("CUDA driver init failed: {e}")))?;

        // Check that the requested device exists.
        let count = Self::count()?;
        if device_id >= count {
            return Err(GpuError::DeviceNotFound(device_id));
        }

        let ctx = CudaContext::new(device_id).map_err(|e| {
            GpuError::CudaError(format!(
                "failed to create CUDA context on device {device_id}: {e}"
            ))
        })?;
        let stream = ctx.default_stream();

        Ok(Self { ctx, stream })
    }

    /// Number of available CUDA devices.
    ///
    /// Initializes the CUDA driver if needed. Returns 0 on machines without
    /// a CUDA-capable GPU (rather than erroring).
    pub fn count() -> Result<usize, GpuError> {
        cudarc::driver::result::init()
            .map_err(|e| GpuError::CudaError(format!("CUDA driver init failed: {e}")))?;

        let n = cudarc::driver::result::device::get_count()
            .map_err(|e| GpuError::CudaError(format!("cuDeviceGetCount failed: {e}")))?;
        Ok(n as usize)
    }

    /// Allocate zero-initialized device memory.
    ///
    /// Allocates `n` elements of type `T` on the GPU, all set to zero.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        n: usize,
    ) -> Result<CudaSlice<T>, GpuError> {
        self.stream
            .alloc_zeros::<T>(n)
            .map_err(|e| GpuError::OutOfMemory(format!("alloc_zeros({n}) failed: {e}")))
    }

    /// Copy host data to device (synchronous).
    ///
    /// Allocates a new device buffer and copies all elements from the host
    /// slice into it.
    pub fn htod_copy<T: DeviceRepr>(&self, data: &[T]) -> Result<CudaSlice<T>, GpuError> {
        self.stream
            .clone_htod(data)
            .map_err(|e| GpuError::CudaError(format!("host-to-device copy failed: {e}")))
    }

    /// Copy device data to host (synchronous).
    ///
    /// Allocates a new `Vec<T>` and copies all elements from the device
    /// buffer into it.
    pub fn dtoh_copy<T: DeviceRepr>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>, GpuError> {
        self.stream
            .clone_dtoh(buf)
            .map_err(|e| GpuError::CudaError(format!("device-to-host copy failed: {e}")))
    }

    /// Load a compiled PTX module into this device's context.
    ///
    /// The returned [`CudaModule`] can be used to look up kernel functions
    /// via `module.load_function("kernel_name")`.
    pub fn load_module(&self, ptx: Ptx) -> Result<Arc<CudaModule>, GpuError> {
        self.ctx
            .load_module(ptx)
            .map_err(|e| GpuError::ModuleLoadError(format!("{e}")))
    }

    /// Access the underlying [`CudaContext`].
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Access the default [`CudaStream`].
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Synchronize the default stream, blocking until all queued GPU work completes.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.stream
            .synchronize()
            .map_err(|e| GpuError::CudaError(format!("stream synchronize: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B4: Test CUDA device enumeration.
    ///
    /// This test verifies that `GpuDevice::count()` successfully queries the
    /// CUDA driver and returns a non-negative count. On machines without a
    /// GPU, the CUDA driver init may fail — the test handles both cases.
    #[test]
    fn test_device_count() {
        match GpuDevice::count() {
            Ok(n) => {
                // On a GPU machine, we expect at least 1 device.
                // On CI without GPUs, count might be 0.
                println!("CUDA devices found: {n}");
                assert!(n <= 256, "unreasonable device count: {n}");
            }
            Err(GpuError::CudaError(msg)) => {
                // Acceptable: CUDA driver not available on this machine.
                println!("CUDA not available (expected on non-GPU machines): {msg}");
            }
            Err(e) => {
                panic!("unexpected error from GpuDevice::count(): {e}");
            }
        }
    }

    /// Verify that requesting an out-of-range device ordinal returns
    /// `DeviceNotFound`.
    #[test]
    fn test_device_not_found() {
        match GpuDevice::new(9999) {
            Err(GpuError::DeviceNotFound(id)) => {
                assert_eq!(id, 9999);
            }
            Err(GpuError::CudaError(_)) => {
                // CUDA not available — acceptable on non-GPU machines.
            }
            Ok(_) => panic!("should not succeed with device 9999"),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
