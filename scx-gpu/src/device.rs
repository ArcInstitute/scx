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

/// Map a cudarc driver error onto the right [`GpuError`] variant, keeping an
/// out-of-memory *typed* as one.
///
/// Every allocating call that is not `alloc_zeros` reaches the driver through a
/// path whose failures all look alike at the Rust level — `clone_htod` returns
/// the same `DriverError` for a bad context and for a card with no memory left.
/// Flattening those to [`GpuError::CudaError`] made `GpuError::OutOfMemory`
/// mean "the allocation went through `alloc_zeros`" rather than "the device is
/// out of memory", which is not a distinction any caller wants to reason about.
///
/// It is load-bearing for [`GpuError::alternate_route_may_succeed`]: that
/// predicate exists to say *an OOM is not worth retrying by another route to
/// the same device*, and it cannot say so about an OOM disguised as a generic
/// CUDA fault. Found by **codex** on PR #422, which traced the disguise from
/// `htod_copy` through to `to_gpu_anndata`'s fallback.
///
/// `CUDA_ERROR_OUT_OF_MEMORY` (code 2) is the only code special-cased; every
/// other driver failure stays a `CudaError`.
fn classify_driver_error(context: &str, e: cudarc::driver::DriverError) -> GpuError {
    if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        GpuError::OutOfMemory(format!("{context}: {e}"))
    } else {
        GpuError::CudaError(format!("{context}: {e}"))
    }
}

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
            .map_err(|e| classify_driver_error("host-to-device copy failed", e))
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
    /// calls — a second `gpu_randomized_pca` / GPU DE
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

    /// Release this process's reserved-but-unused VRAM from the CUDA default
    /// async memory pool back to the driver.
    ///
    /// cudarc 0.19's default stream allocates via `cuMemAllocAsync` and frees
    /// via `cuMemFreeAsync` on pool-capable GPUs ([`CudaContext::has_async_alloc`]).
    /// Freed allocations return to the *async pool*, which retains the
    /// high-water-mark physical pages for the whole process — and a synchronous
    /// `cudaMalloc` allocator (cupy / RMM / rapids-singlecell) cannot draw from
    /// those pages. So after a large native op (or one that OOMs), the pool can
    /// hold tens of GB that a subsequent rapids op then fails to allocate
    /// (report B9). Trimming the pool to 0 returns that VRAM to the driver.
    ///
    /// Synchronizes first so the queued `cuMemFreeAsync`s have completed (and
    /// thus count as "unused"); `cuMemPoolTrimTo` only frees memory **not
    /// currently in use**, so live allocations (e.g. a `to_gpu_anndata` handoff
    /// cuPy still references) are untouched. No-op when the device has no async
    /// memory pool.
    ///
    /// Invoked from [`Drop`], i.e. once per `GpuDevice` lifetime. This assumes
    /// the established usage pattern — **one `GpuDevice` per accelerator op**,
    /// never cached across many small ops — so the trim runs at op boundaries.
    /// A caller that holds a `GpuDevice` across a hot loop of tiny ops would
    /// instead pay a full device `synchronize()` + lose the async pool's
    /// allocation caching on every drop; cache the matrices/handles, not the
    /// device, in that case.
    pub fn reclaim_memory_pool(&self) -> Result<(), GpuError> {
        if !self.ctx.has_async_alloc() {
            return Ok(());
        }
        self.synchronize()?;
        // SAFETY: `ordinal` is this context's device ordinal, so `device::get`
        // returns a valid CUdevice; `get_default_mem_pool` returns the driver-
        // owned default pool for it; `trim_to` only releases unused memory.
        unsafe {
            let dev = cudarc::driver::result::device::get(self.ctx.ordinal() as i32)
                .map_err(|e| GpuError::CudaError(format!("cuDeviceGet failed: {e}")))?;
            let pool = cudarc::driver::result::device::get_default_mem_pool(dev).map_err(|e| {
                GpuError::CudaError(format!("cuDeviceGetDefaultMemPool failed: {e}"))
            })?;
            cudarc::driver::result::mem_pool::trim_to(pool, 0)
                .map_err(|e| GpuError::CudaError(format!("cuMemPoolTrimTo failed: {e}")))?;
        }
        Ok(())
    }
}

impl Drop for GpuDevice {
    /// Return reserved async-pool VRAM to the driver when the per-op device is
    /// dropped — on both the success and error/OOM paths — so one op cannot
    /// strand VRAM for the rest of the process (report B9). Best-effort: a
    /// failed trim is non-fatal and must never panic in `Drop`.
    fn drop(&mut self) {
        let _ = self.reclaim_memory_pool();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An out-of-memory driver error must stay typed as one no matter which
    /// cudarc call produced it.
    ///
    /// `DriverError` is a public tuple over `CUresult`, so this exercises the
    /// real mapping on a CPU host — no device needed, and no reliance on
    /// `GpuError::OutOfMemory` being constructed by hand, which is exactly the
    /// gap that let `htod_copy` report an OOM as a generic `CudaError`.
    #[test]
    fn a_driver_oom_is_classified_as_out_of_memory() {
        let e = classify_driver_error(
            "host-to-device copy failed",
            cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY),
        );
        assert!(
            matches!(e, GpuError::OutOfMemory(_)),
            "expected OutOfMemory, got {e}"
        );
        // And therefore no alternate route to the same device is worth trying.
        assert!(!e.alternate_route_may_succeed());
        assert!(e.is_runtime_failure());
    }

    #[test]
    fn a_non_oom_driver_error_stays_a_cuda_error() {
        let e = classify_driver_error(
            "host-to-device copy failed",
            cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE),
        );
        assert!(matches!(e, GpuError::CudaError(_)), "got {e}");
        // A generic driver fault is still worth another route (that is what
        // makes the OOM carve-out meaningful rather than vacuous).
        assert!(e.alternate_route_may_succeed());
    }

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
