//! GPU device management wrapping cudarc 0.19 `CudaContext` + `CudaStream`.
//!
//! `GpuDevice` provides a thin ergonomic wrapper over cudarc's low-level
//! CUDA driver API, handling context creation, stream management, memory
//! operations, and module loading. All CUDA errors are mapped to [`GpuError`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::safe::{
    CudaContext, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, HostSlice,
    ValidAsZeroBits,
};
use cudarc::driver::LaunchConfig;
use cudarc::nvrtc::Ptx;

use crate::capture_guard;
use crate::error::GpuError;

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
/// `CUDA_ERROR_OUT_OF_MEMORY` is the only code special-cased; every other driver
/// failure stays a `CudaError`.
fn classify_driver_error(context: &str, e: cudarc::driver::DriverError) -> GpuError {
    // `e.to_string()` is what forces the driver library to load: `DriverError`'s
    // Display calls `cuGetErrorString`. Rendering here, and passing the finished
    // string down, is what lets the decision be tested on a host with no CUDA
    // driver at all — see `classify_driver_status`.
    classify_driver_status(context, e.0 as u32, &e.to_string())
}

/// The driver status code for an out-of-memory failure.
///
/// Spelled as a literal rather than read from `cudarc::driver::sys` at run time,
/// because *touching* that enum's `Display` path dlopens `libcuda`. The
/// `const _` below pins the literal to cudarc's own value at **compile** time,
/// so a renumbering upstream is a build error on every host — including the
/// driverless CI runners, which is exactly where a runtime lookup could not go.
const CUDA_ERROR_OUT_OF_MEMORY_CODE: u32 = 2;

const _: () = assert!(
    cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32 == CUDA_ERROR_OUT_OF_MEMORY_CODE,
    "cudarc renumbered CUDA_ERROR_OUT_OF_MEMORY; update CUDA_ERROR_OUT_OF_MEMORY_CODE"
);

/// The classification decision, over a plain status code and an already-rendered
/// message.
///
/// Split out from [`classify_driver_error`] so it can be tested without a CUDA
/// driver present. The first version of this test constructed a synthetic
/// `DriverError` and passed locally — on a host that happens to have `libcuda`
/// — while panicking on CI, because formatting the error is what loads the
/// library. A test for a pure decision should not need a driver, and now does
/// not.
fn classify_driver_status(context: &str, code: u32, rendered: &str) -> GpuError {
    if code == CUDA_ERROR_OUT_OF_MEMORY_CODE {
        GpuError::OutOfMemory(format!("{context}: {rendered}"))
    } else {
        GpuError::CudaError(format!("{context}: {rendered}"))
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

/// First line of the placeholder PTX `build.rs` writes when `nvcc` is absent.
///
/// Byte-identical to the literal in `scx-gpu/build.rs`;
/// [`tests::the_stub_marker_matches_the_one_build_rs_writes`] reads that file
/// and asserts it, because two spellings of one sentinel is the whole failure
/// mode this closes.
pub const PTX_STUB_MARKER: &str = "// SCX_PTX_IS_STUB";

/// Refuse a placeholder PTX at load time, with the cause and the fix.
///
/// Without this, a stub build fails at the *first kernel launch* as
/// `CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid` or
/// `named symbol not found` — a runtime symptom four steps from its cause, and
/// one that reads as a broken kernel rather than a broken build. Review §8.15.
///
/// It is reachable, not hypothetical: `build.rs` re-runs on `.cu` changes, and
/// before this commit neither PATH nor CUDA_HOME was in its fingerprint. A
/// target directory that once saw a CPU-only build kept replaying the stub
/// branch on a machine that has `nvcc` — the Phase 7 gate burned a GPU
/// allocation on exactly that, dying in 82 s.
///
/// A prefix test, not a `contains`: the marker is the stub's first line, and a
/// real `nvcc` module can be megabytes.
fn reject_ptx_stub(ptx_src: &str) -> Result<(), GpuError> {
    if !ptx_src.starts_with(PTX_STUB_MARKER) {
        return Ok(());
    }
    Err(GpuError::ModuleLoadError(format!(
        "this build has placeholder PTX, not compiled kernels: the module \
         begins with `{PTX_STUB_MARKER}`, which scx-gpu's build script writes \
         when `nvcc` is not on PATH. No GPU kernel in this binary can run. \
         Rebuild with the CUDA toolkit on PATH (a conda env with `cuda-nvcc`, \
         or /usr/local/cuda/bin) — and if the rebuild is a no-op, clear \
         CARGO_TARGET_DIR wholesale, because the stub was cached from an \
         earlier build on a machine without nvcc. Set SCX_GPU_REQUIRE_NVCC=1 \
         to turn this into a build-time failure instead."
    )))
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
        capture_guard::check("GpuDevice::alloc_zeros")?;
        self.stream
            .alloc_zeros::<T>(n)
            .map_err(|e| GpuError::OutOfMemory(format!("alloc_zeros({n}) failed: {e}")))
    }

    /// Allocate zero-initialized device memory **on a caller-supplied stream**.
    ///
    /// [`Self::alloc_zeros`] allocates on this device's own stream. A few
    /// callers must allocate on the stream that will use the buffer instead —
    /// `staging.rs`'s slot grow, whose buffers are then read by kernels on the
    /// caller's stream, and where allocating elsewhere makes cudarc insert a
    /// cross-stream wait.
    ///
    /// This exists so that those callers do not reach past `GpuDevice` to
    /// `stream.alloc_zeros` directly. Doing so used to be the one production
    /// allocation the CUDA-graph capture guard could not see; routing it here
    /// makes `GpuDevice` the crate's sole allocation funnel, which is what lets
    /// one [`capture_guard::check`] cover all of it. A CI guard rejects any new
    /// bare `.alloc_zeros(` outside this file.
    pub fn alloc_zeros_on<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        stream: &Arc<CudaStream>,
        n: usize,
    ) -> Result<CudaSlice<T>, GpuError> {
        capture_guard::check("GpuDevice::alloc_zeros_on")?;
        stream
            .alloc_zeros::<T>(n)
            .map_err(|e| GpuError::OutOfMemory(format!("alloc_zeros_on({n}) failed: {e}")))
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
        capture_guard::check("GpuDevice::htod_copy")?;
        self.stream
            .clone_htod(data)
            .map_err(|e| classify_driver_error("host-to-device copy failed", e))
    }

    /// Copy device data to host (synchronous).
    ///
    /// Allocates a new `Vec<T>` and copies all elements from the device
    /// buffer into it.
    pub fn dtoh_copy<T: DeviceRepr>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>, GpuError> {
        capture_guard::check("GpuDevice::dtoh_copy")?;
        self.stream
            .clone_dtoh(buf)
            .map_err(|e| GpuError::CudaError(format!("device-to-host copy failed: {e}")))
    }

    /// Load a compiled PTX module into this device's context.
    ///
    /// The returned [`CudaModule`] can be used to look up kernel functions
    /// via `module.load_function("kernel_name")`.
    pub fn load_module(&self, ptx: Ptx) -> Result<Arc<CudaModule>, GpuError> {
        capture_guard::check("GpuDevice::load_module")?;
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
        reject_ptx_stub(ptx_src)?;
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

    /// Clone this device with a different default stream.
    /// Used by CUDA-Graph capture sites that need kernel
    /// launches to flow through a capturable stream
    /// (e.g. `ctx.per_thread_stream()`) without changing every kernel
    /// function's signature.
    ///
    /// The shared `CudaContext` is reference-counted, and the module
    /// cache is shallow-cloned — `Arc<CudaModule>` entries shared with
    /// the original keep the GPU-side module load amortized.
    pub fn with_stream(&self, stream: Arc<CudaStream>) -> Self {
        Self {
            ctx: self.ctx.clone(),
            stream,
            module_cache: RefCell::new(self.module_cache.borrow().clone()),
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
        capture_guard::check("GpuDevice::synchronize")?;
        self.stream
            .synchronize()
            .map_err(|e| GpuError::CudaError(format!("stream synchronize: {e}")))
    }

    /// Copy device data to host from any device pointer, on a caller-supplied
    /// stream, returning a fresh `Vec`.
    ///
    /// [`Self::dtoh_copy`] takes a whole `CudaSlice` on this device's own
    /// stream. Nine sites in `scx-accel/src/diffexp/gpu.rs` read back a **view**
    /// (`&d_sums.slice(..)`) on a chunk stream instead, so they need the generic
    /// form.
    ///
    /// This is a **host sync**, and that is why it is funnelled: `clone_dtoh`
    /// blocks the calling thread until the copy completes, which is
    /// capture-illegal exactly like an allocation. Missing this family is what
    /// **codex** and **Cursor Agent** both flagged on PR #473 — the funnel
    /// checked method spellings (`alloc_zeros`, `synchronize`) rather than the
    /// operation families CUDA prohibits during capture.
    pub fn clone_dtoh_from<T: DeviceRepr, S: DevicePtr<T>>(
        &self,
        stream: &Arc<CudaStream>,
        src: &S,
    ) -> Result<Vec<T>, GpuError> {
        capture_guard::check("GpuDevice::clone_dtoh_from")?;
        stream
            .clone_dtoh(src)
            .map_err(|e| GpuError::CudaError(format!("device-to-host copy failed: {e}")))
    }

    /// Copy device data into an existing host buffer, on a caller-supplied
    /// stream. The in-place sibling of [`Self::clone_dtoh_from`], and a host
    /// sync for the same reason — see that method.
    pub fn memcpy_dtoh_into<T: DeviceRepr, S: DevicePtr<T>, D: HostSlice<T> + ?Sized>(
        &self,
        stream: &Arc<CudaStream>,
        src: &S,
        dst: &mut D,
    ) -> Result<(), GpuError> {
        capture_guard::check("GpuDevice::memcpy_dtoh_into")?;
        stream
            .memcpy_dtoh(src, dst)
            .map_err(|e| GpuError::CudaError(format!("device-to-host copy failed: {e}")))
    }

    /// Block until all work queued on a **caller-supplied** stream completes.
    ///
    /// [`Self::synchronize`] waits on this device's own stream. Three callers
    /// wait on a stream they were handed instead: `cusparse`'s profiling
    /// timer, whose measurement is meaningless without it; `nvcomp`'s
    /// decompress, which must not drop its staging buffers with a DMA in
    /// flight; and `shufdelta_gpu`'s error path, draining a copy stream before
    /// pinned host buffers go out of scope.
    ///
    /// Routed here for the same reason as [`Self::alloc_zeros_on`]: a host sync
    /// inside a CUDA-graph capture region is as illegal as an allocation, and a
    /// raw `stream.synchronize()` is invisible to [`capture_guard`]. A CI guard
    /// rejects any new `.synchronize()` on a stream receiver outside this file.
    ///
    /// Note the boundary: `CudaEvent::synchronize` is *not* funnelled through
    /// here — the shard sources wait on per-slot events, and an event is not a
    /// stream. Those sites are outside any capture region today; if one ever
    /// moves inside, this is where the equivalent helper belongs.
    pub fn synchronize_stream(&self, stream: &CudaStream) -> Result<(), GpuError> {
        capture_guard::check("GpuDevice::synchronize_stream")?;
        stream
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
        // Checked even though `Drop` discards the result, and that combination
        // is the point: a `GpuDevice` dropped inside a capture region used to
        // reach `self.synchronize()` below and invalidate the capture. It now
        // declines instead, and the trim happens at the next drop outside one.
        capture_guard::check("GpuDevice::reclaim_memory_pool")?;
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

/// 1-D launch geometry for a flat, one-thread-per-element kernel.
///
/// `total` is counted in `u64` — never `usize as u32` — so the grid covers every
/// element however large the matrix, and the block count is checked against the
/// CUDA `grid_dim.x` cap **before** the cast, so an unlaunchable grid errors
/// instead of wrapping into a plausible-looking small one.
///
/// Note the two thresholds are far apart, and neither is 2³¹ *elements*: the
/// `u64` count is what fixes the >2³² truncation, while the cap rejection needs
/// `blocks > i32::MAX`, i.e. `total > 256 × (2³¹ − 1) ≈ 5.5e11` elements at the
/// usual 256-thread block. A 2³¹-element matrix launches normally, as
/// `a_flat_grid_covers_every_element_past_2_31` asserts.
///
/// This exists because the truncating form is invisible: `(total as u32)` on a
/// matrix of 2³² + 1000 elements yields a grid four blocks wide, the kernel
/// returns having touched the first thousand elements, and the caller gets a
/// partly-transformed matrix with no error to attribute it to (review §8.4).
///
/// This is only the host half. The kernel's own element count and flat index
/// must be 64-bit too — a 32-bit `int total = m * k` overflows a step earlier,
/// at 2³¹, and being signed overflow it is UB rather than a defined wrap — so
/// widening the grid alone buys nothing. See
/// `kernels/colmajor_ops.cu`'s `mean_correct_colmajor_strided_kernel` for the
/// reference shape.
///
/// `total == 0` resolves to an empty grid rather than an error; callers still
/// short-circuit earlier to skip the module load.
///
/// # Errors
///
/// [`GpuError::ShapeMismatch`] when `total / threads` exceeds `i32::MAX`, the
/// driver's maximum `grid_dim.x`, or when `threads` is zero. The zero check is
/// a real check rather than a `debug_assert!`: this returns a `Result` and is
/// re-exported from the crate root, and in a release build the assert would
/// vanish and leave `div_ceil` to panic with a bare divide-by-zero instead.
pub fn flat_launch_1d(total: u64, threads: u32) -> Result<LaunchConfig, GpuError> {
    if threads == 0 {
        return Err(GpuError::ShapeMismatch {
            expected: "threads > 0 (CUDA block_dim.x)".into(),
            got: "threads = 0".into(),
        });
    }
    let blocks = total.div_ceil(threads as u64);
    if blocks > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: format!("total / {threads} <= 2^31 - 1 (CUDA grid_dim.x cap)"),
            got: format!("total = {total} needs {blocks} blocks"),
        });
    }
    Ok(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    /// The sentinel `build.rs` writes and the one `device.rs` checks must be
    /// the same bytes. They are declared in two files that cannot import each
    /// other, so the only thing standing between them is this test — and two
    /// spellings of one sentinel is precisely the failure this guard exists to
    /// prevent: the check would silently never fire and a stub build would go
    /// back to dying at the first kernel launch.
    #[test]
    fn the_stub_marker_matches_the_one_build_rs_writes() {
        let build_rs = include_str!("../build.rs");
        assert!(
            build_rs.contains(super::PTX_STUB_MARKER),
            "build.rs no longer writes `{}` — the load-time stub check in \
             device.rs can never fire (review §8.15)",
            super::PTX_STUB_MARKER
        );
    }

    /// A stub is refused, and the message says what to do about it. Runs on
    /// CPU: the check is a string test on the module source, which is the
    /// reason it was put at that layer rather than after `Ptx::from_src`.
    #[test]
    fn a_stub_module_is_refused_with_an_actionable_message() {
        let stub = "// SCX_PTX_IS_STUB — nvcc not available at build time\n";
        let err = super::reject_ptx_stub(stub).expect_err("a stub must be refused");
        let msg = err.to_string();
        for want in ["placeholder PTX", "nvcc", "CARGO_TARGET_DIR"] {
            assert!(msg.contains(want), "message lacks `{want}`: {msg}");
        }
    }

    /// The accept side. A guard with only a reject test can be satisfied by
    /// refusing everything, which would take every GPU op down.
    #[test]
    fn real_ptx_is_not_mistaken_for_a_stub() {
        // The shape nvcc actually emits: a comment banner, then directives.
        let real = "//\n// Generated by NVIDIA NVVM Compiler\n//\n.version 7.0\n";
        assert!(super::reject_ptx_stub(real).is_ok());
        // And a module that merely *mentions* the marker further in is fine —
        // the check is anchored at the start for exactly this reason.
        let mentions = "//\n// see SCX_PTX_IS_STUB\n.version 7.0\n";
        assert!(super::reject_ptx_stub(mentions).is_ok());
    }

    use super::*;

    /// The grid a flat kernel is launched with must cover **every** element.
    ///
    /// The failure this pins is not a crash: `(total as u32).div_ceil(threads)`
    /// truncates the element count, the kernel is handed far too few threads,
    /// and the tail of the matrix is simply never visited — no error anywhere.
    /// It bites past 2³², so the first two cases below pass either way and are
    /// here to bound the regression from the other side; the last two are what
    /// goes red against the truncating form. `36M × 60` is a real census-scale
    /// PCA shape (`n_obs × k`), just under the u32 boundary.
    #[test]
    fn a_flat_grid_covers_every_element_past_2_31() {
        for total in [
            (1u64 << 31) + 1,              // past the signed-int wrap
            36_000_000u64 * 60,            // 36 M cells × k=60 = 2.16e9
            (1u64 << 32) + 1000,           // past the u32 wrap
            (u32::MAX as u64 + 1) * 3 / 2, // 1.5 × u32::MAX
        ] {
            let threads = 256u32;
            let cfg = flat_launch_1d(total, threads).expect("geometry must resolve");
            let covered = cfg.grid_dim.0 as u64 * cfg.block_dim.0 as u64;
            assert!(
                covered >= total,
                "grid covers {covered} threads but {total} elements need visiting",
            );
        }
    }

    /// Past the CUDA `grid_dim.x` cap the answer is an error, never a grid that
    /// silently wrapped into range.
    #[test]
    fn a_flat_grid_past_the_grid_dim_cap_is_rejected() {
        let threads = 256u32;
        let total = (i32::MAX as u64 + 1) * threads as u64;
        let err = flat_launch_1d(total, threads).expect_err("must not launch past the cap");
        assert!(
            matches!(err, GpuError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err}"
        );
    }

    /// Zero elements resolve to an empty grid rather than an error — the
    /// wrappers all short-circuit before calling, and this keeps the helper
    /// honest for any that forget to.
    #[test]
    fn a_flat_grid_of_zero_elements_is_empty() {
        let cfg = flat_launch_1d(0, 256).expect("zero is not an error");
        assert_eq!(cfg.grid_dim.0, 0);
    }

    /// A zero *block size* is an `Err`, not a panic — the check has to survive
    /// a release build, where a `debug_assert!` would be compiled out and
    /// `div_ceil` would divide by zero instead. Found by **gemini-code-assist**.
    #[test]
    fn a_zero_block_size_is_rejected_not_panicked_on() {
        let err = flat_launch_1d(1024, 0).expect_err("zero threads must not launch");
        assert!(
            matches!(err, GpuError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err}"
        );
    }

    /// An out-of-memory driver failure must stay typed as one no matter which
    /// cudarc call produced it.
    ///
    /// Drives `classify_driver_status` with the raw code rather than building a
    /// `DriverError`: formatting one calls `cuGetErrorString`, which dlopens
    /// `libcuda`. The first version of this test did build one, passed here,
    /// and panicked on CI's driverless runner. The code being a real
    /// `CUDA_ERROR_OUT_OF_MEMORY` is pinned by the `const _` assertion beside
    /// the constant — by the compiler, on every host, rather than by this test.
    #[test]
    fn a_driver_oom_is_classified_as_out_of_memory() {
        let e = classify_driver_status(
            "host-to-device copy failed",
            CUDA_ERROR_OUT_OF_MEMORY_CODE,
            "out of memory",
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
        // 1 == CUDA_ERROR_INVALID_VALUE. Any non-OOM code will do; the point is
        // that the carve-out is narrow rather than swallowing everything.
        let e = classify_driver_status("host-to-device copy failed", 1, "invalid argument");
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
