//! GPU device management wrapping cudarc 0.19 `CudaContext` + `CudaStream`.
//!
//! `GpuDevice` provides a thin ergonomic wrapper over cudarc's low-level
//! CUDA driver API, handling context creation, stream management, memory
//! operations, and module loading. All CUDA errors are mapped to [`GpuError`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::safe::{
    CudaContext, CudaModule, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use crate::error::GpuError;
use crate::gpu_graph::GpuGraphCache;

/// A GPU device handle wrapping a CUDA context and its default stream.
///
/// In cudarc 0.19, `CudaDevice` was removed. The primary abstractions are now
/// [`CudaContext`] (device handle) and [`CudaStream`] (work scheduling).
/// Memory operations (`alloc_zeros`, `clone_htod`, `clone_dtoh`) are methods
/// on `CudaStream` via `&Arc<Self>`, not the context.
pub struct GpuDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Cache of loaded PTX modules, keyed by PTX source string pointer.
    /// Avoids re-parsing and re-loading the same PTX on every kernel call.
    module_cache: RefCell<HashMap<*const str, Arc<CudaModule>>>,
    /// Cache of captured `cudaGraph_t` keyed by shape signature
    /// ([`crate::gpu_graph::GraphKey`]). Iteration-heavy stages (PCA
    /// power, Harmony k-means, UMAP SGD, GPU DE chunk loops) capture
    /// their stable kernel sequence once and replay on subsequent
    /// iterations to amortize per-launch latency.
    graph_cache: RefCell<GpuGraphCache>,
}

/// Safely initialize the CUDA driver, catching panics from cudarc when
/// `libcuda.so` is not available (e.g., on CI runners without a GPU).
fn safe_cuda_init() -> Result<(), GpuError> {
    std::panic::catch_unwind(cudarc::driver::result::init)
        .map_err(|_| {
            GpuError::CudaError("CUDA driver not available (libcuda.so not found)".into())
        })?
        .map_err(|e| GpuError::CudaError(format!("CUDA driver init failed: {e}")))
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
        safe_cuda_init()?;

        // Check that the requested device exists.
        let count = Self::count()?;
        if device_id >= count {
            return Err(GpuError::DeviceNotFound(device_id));
        }

        let ctx = std::panic::catch_unwind(|| CudaContext::new(device_id))
            .map_err(|_| {
                GpuError::CudaError(format!(
                    "CUDA context creation panicked on device {device_id}"
                ))
            })?
            .map_err(|e| {
                GpuError::CudaError(format!(
                    "failed to create CUDA context on device {device_id}: {e}"
                ))
            })?;
        let stream = ctx.default_stream();

        Ok(Self {
            ctx,
            stream,
            module_cache: RefCell::new(HashMap::new()),
            graph_cache: RefCell::new(GpuGraphCache::new()),
        })
    }

    /// Number of available CUDA devices.
    ///
    /// Initializes the CUDA driver if needed. Returns 0 on machines without
    /// a CUDA-capable GPU (rather than erroring).
    pub fn count() -> Result<usize, GpuError> {
        safe_cuda_init()?;

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

    /// Query free and total GPU memory in bytes.
    ///
    /// Returns `(free, total)` via `cuMemGetInfo_v2`. Useful for pre-flight
    /// checks before large allocations (e.g., GPU PCA).
    pub fn free_memory(&self) -> Result<(usize, usize), GpuError> {
        self.ctx
            .mem_get_info()
            .map_err(|e| GpuError::CudaError(format!("cuMemGetInfo failed: {e}")))
    }

    /// Maximum opt-in dynamic shared memory per block, in bytes.
    ///
    /// Returns `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` — the
    /// upper bound a kernel may request via
    /// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, …)`.
    /// Distinct from the default per-block ceiling
    /// (`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`, typically 48 KB):
    /// crossing the default requires per-function opt-in. On H100 (CC 9.0)
    /// this returns ~100 KB; on older devices that don't support opt-in it
    /// falls back to the default ceiling.
    ///
    /// Used by `gpu_de_pseudobulk_csc_direct` to decide between default-SMEM
    /// launch, opt-in launch, or adaptive `blockDim.x` when `n_groups` is
    /// large enough that `n_groups × blockDim.x × sizeof(f64)` would exceed
    /// the default 48 KB.
    pub fn max_dynamic_shared_mem_per_block(&self) -> Result<usize, GpuError> {
        let v = self
            .ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|e| {
                GpuError::CudaError(format!(
                    "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) failed: {e}"
                ))
            })?;
        Ok(v.max(0) as usize)
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

    /// Load a PTX module, returning a cached copy if the same source was loaded before.
    ///
    /// The cache key is the pointer identity of the `&'static str` PTX source,
    /// so this works correctly with `include_str!()` constants (each constant has
    /// a unique address). Repeated calls with the same PTX source skip the
    /// CUDA JIT compilation entirely.
    pub fn load_module_cached(&self, ptx_src: &'static str) -> Result<Arc<CudaModule>, GpuError> {
        let key = ptx_src as *const str;
        if let Some(module) = self.module_cache.borrow().get(&key) {
            return Ok(Arc::clone(module));
        }
        let ptx = Ptx::from_src(ptx_src);
        let module = self.load_module(ptx)?;
        self.module_cache
            .borrow_mut()
            .insert(key, Arc::clone(&module));
        Ok(module)
    }

    /// Access the underlying [`CudaContext`].
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Access the default [`CudaStream`].
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Borrow the per-device CUDA Graph cache. The cache persists across
    /// calls — a second `gpu_randomized_pca` / `pdex_ref_gpu_chunked`
    /// run with the same shape signature replays the cached graph
    /// rather than recapturing.
    pub fn graph_cache(&self) -> std::cell::RefMut<'_, GpuGraphCache> {
        self.graph_cache.borrow_mut()
    }

    /// Clone this device with a different default stream.
    /// Used by CUDA-Graph capture sites that need kernel
    /// launches to flow through a capturable stream
    /// (e.g. `ctx.per_thread_stream()`) without changing every kernel
    /// function's signature.
    ///
    /// The shared `CudaContext` is reference-counted, and the module
    /// cache is shallow-cloned — `Arc<CudaModule>` entries shared with
    /// the original keep the GPU-side module load amortized. The clone
    /// gets a fresh, empty `GpuGraphCache`; callers that want to share
    /// graph entries across stream variants must currently route them
    /// through the original device's cache.
    pub fn with_stream(&self, stream: Arc<CudaStream>) -> Self {
        Self {
            ctx: self.ctx.clone(),
            stream,
            module_cache: RefCell::new(self.module_cache.borrow().clone()),
            graph_cache: RefCell::new(GpuGraphCache::new()),
        }
    }

    /// Query the human-readable name of the GPU device (e.g. "NVIDIA A100-SXM4-80GB").
    pub fn name(&self) -> Result<String, GpuError> {
        self.ctx
            .name()
            .map_err(|e| GpuError::CudaError(format!("cuDeviceGetName failed: {e}")))
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
