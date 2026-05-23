//! cuSPARSE CSR interop for GPU-resident sparse matrices.
//!
//! Provides [`CusparseHandle`] (library handle), methods on [`GpuCsr`] to
//! create cuSPARSE sparse matrix descriptors (`cusparseSpMatDescr_t`),
//! [`DnMatDescr`] for dense matrix descriptors, and SpMM (sparse × dense
//! matrix multiply) wrappers for GPU-accelerated PCA.
//!
//! ## SpMM Usage
//!
//! ```ignore
//! let handle = CusparseHandle::new()?;
//! let a_desc = gpu_csr.to_cusparse_csr(&dev, dev.stream())?;
//! // B is column-major dense (k × n), C is column-major dense (m × n)
//! spmm_csr(&handle, dev.stream(), &a_desc, &b_device, &mut c_device,
//!          m, k, n, 1.0, 0.0)?;
//! ```

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::cusparse::sys::{
    self as csp, cudaDataType, cusparseIndexBase_t, cusparseIndexType_t, cusparseSpMatDescr_t,
};
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::cast_gpu::cast_i64_to_i32_gpu;
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_decode::GpuCsr;

/// Cached cuSPARSE ABI probe — `true` when the runtime libcusparse satisfies
/// cudarc 0.19+ expectations, `false` when it predates cuSPARSE 12.5 and is
/// missing required symbols. See `cusparse_modern_abi_available`.
static CUSPARSE_MODERN_ABI: OnceLock<bool> = OnceLock::new();

/// Probe the runtime `libcusparse.so` for `cusparseBsrSetStridedBatch` — a
/// canary symbol added in cuSPARSE 12.5 (CUDA Toolkit 12.5, mid-2024) and
/// expected by cudarc 0.19+. Returns `false` when the library is older than
/// 12.5, so callers can route to a CPU fallback instead of letting cudarc's
/// lazy `dlsym` panic deep inside an FFI call.
///
/// On Ubuntu hosts the system
/// `libcusparse-dev` ships 12.0.1.140 (Jan 2023) at
/// `/usr/lib/x86_64-linux-gnu/libcusparse.so`. If the toolkit's newer
/// libcusparse at `/usr/local/cuda*/lib64/libcusparse.so` is not earlier on
/// `LD_LIBRARY_PATH`, the loader picks the system version and `cusparseCreate`
/// panics with `undefined symbol: cusparseBsrSetStridedBatch`.
///
/// The first call dlopens libcusparse via `libloading`; subsequent calls
/// return the cached boolean without re-probing. We try the unversioned
/// `libcusparse.so` first — matching cudarc 0.19's single load path via
/// `libloading::library_filename("cusparse")` — and then a small fixed list
/// of versioned SONAMEs as fallbacks for hosts where only the versioned file
/// is on the loader path (libcusparse-dev not installed, conda-only layouts,
/// some container images). The candidate list is a strict superset of
/// cudarc's: `.so.12` is the current CUDA 12.x SONAME, `.so.13` is the
/// expected CUDA 13.x SONAME (forward-compat), and `.so.0` covers
/// occasional legacy/symlink layouts. If cudarc could load cuSPARSE on
/// this host, at least one of these candidates will resolve.
pub fn cusparse_modern_abi_available() -> bool {
    *CUSPARSE_MODERN_ABI.get_or_init(|| {
        let candidates = [
            "libcusparse.so",
            "libcusparse.so.12",
            "libcusparse.so.13",
            "libcusparse.so.0",
        ];
        for name in candidates {
            // SAFETY: dlopen of a system library by name; not unsafe in the
            // memory-safety sense, but the API is `unsafe` because the library
            // can run arbitrary `_init` code. We only probe symbols.
            let lib = match unsafe { libloading::Library::new(name) } {
                Ok(l) => l,
                Err(_) => continue,
            };
            // SAFETY: looking up a symbol by name. Not invoking it.
            let probe: Result<libloading::Symbol<'_, unsafe extern "C" fn()>, _> =
                unsafe { lib.get(b"cusparseBsrSetStridedBatch\0") };
            return probe.is_ok();
        }
        false
    })
}

/// RAII wrapper around a cuSPARSE library handle (`cusparseHandle_t`).
///
/// Created once per device, reusable across multiple SpMV/SpMM calls.
pub struct CusparseHandle {
    raw: csp::cusparseHandle_t,
}

impl CusparseHandle {
    /// Create a new cuSPARSE handle on the current CUDA context.
    pub fn new() -> Result<Self, GpuError> {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            csp::cusparseCreate(handle.as_mut_ptr())
                .result()
                .map_err(|e| GpuError::CuSparseError(format!("cusparseCreate: {e:?}")))?;
            Ok(Self {
                raw: handle.assume_init(),
            })
        }
    }

    /// Access the raw `cusparseHandle_t` for FFI calls.
    pub fn raw(&self) -> csp::cusparseHandle_t {
        self.raw
    }

    /// Bind this handle to the given CUDA stream.
    ///
    /// All subsequent cuSPARSE operations using this handle will execute on
    /// `stream`. Must be called before `spmm_csr` / `spmm_csr_transpose`.
    fn set_stream(&self, stream: &CudaStream) -> Result<(), GpuError> {
        unsafe {
            csp::cusparseSetStream(self.raw, stream.cu_stream() as _)
                .result()
                .map_err(|e| GpuError::CuSparseError(format!("cusparseSetStream: {e:?}")))
        }
    }
}

impl Drop for CusparseHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = csp::cusparseDestroy(self.raw);
        }
    }
}

// SAFETY: cuSPARSE handles are thread-safe per NVIDIA documentation.
// The handle internally synchronizes access to shared state.
unsafe impl Send for CusparseHandle {}
unsafe impl Sync for CusparseHandle {}

/// RAII wrapper around a cuSPARSE sparse matrix descriptor (`cusparseSpMatDescr_t`).
///
/// Created from [`GpuCsr::to_cusparse_csr`]. The descriptor is a lightweight
/// handle that references the existing GPU memory in the `GpuCsr` — no data
/// is copied. The `GpuCsr` must outlive this descriptor.
pub struct CusparseSpMatDescr {
    raw: cusparseSpMatDescr_t,
    /// Downcast i32 indptr buffer owned by this descriptor — cuSPARSE
    /// captured raw pointers inside `raw`, so the buffer must live as long
    /// as the descriptor. Kept private; only [`to_cusparse_csr`] creates it.
    /// See `to_cusparse_csr` docs for why this downcast is needed.
    #[allow(dead_code)]
    _i32_indptr: Option<CudaSlice<i32>>,
}

impl CusparseSpMatDescr {
    /// Access the raw `cusparseSpMatDescr_t` for FFI calls.
    pub fn raw(&self) -> cusparseSpMatDescr_t {
        self.raw
    }

    /// Construct from a raw descriptor and an owned downcast `i32` indptr
    /// buffer. The descriptor must already reference the buffer's device
    /// pointer; this constructor only assumes ownership for lifetime
    /// extension.
    ///
    /// Used by [`crate::staging::GpuCsrSlot`] to build a cached descriptor
    /// over a slot's exact-sized buffer views without going through
    /// [`crate::shard_decode::GpuCsr::to_cusparse_csr`] (which requires an
    /// owned `GpuCsr`).
    pub(crate) fn from_raw_with_i32_indptr(
        raw: cusparseSpMatDescr_t,
        i32_indptr: CudaSlice<i32>,
    ) -> Self {
        Self {
            raw,
            _i32_indptr: Some(i32_indptr),
        }
    }
}

impl Drop for CusparseSpMatDescr {
    fn drop(&mut self) {
        unsafe {
            let _ = csp::cusparseDestroySpMat(self.raw);
        }
    }
}

// SAFETY: cusparseSpMatDescr_t is a lightweight handle that references GPU
// memory. It can safely be sent between threads (the underlying GPU buffers
// handle synchronization via CUDA events).
unsafe impl Send for CusparseSpMatDescr {}

/// RAII wrapper around a cuSPARSE dense matrix descriptor (`cusparseDnMatDescr_t`).
///
/// Wraps `cusparseCreateDnMat` / `cusparseDestroyDnMat`. The descriptor is a
/// lightweight handle that references existing GPU memory — no data is copied.
/// The underlying `CudaSlice` must outlive this descriptor.
pub struct DnMatDescr {
    raw: csp::cusparseDnMatDescr_t,
}

impl DnMatDescr {
    /// Create a dense matrix descriptor for a mutable device buffer.
    ///
    /// The buffer is interpreted as a column-major matrix with dimensions
    /// `rows × cols` and leading dimension `ld` (typically `rows` for col-major).
    pub fn new(
        values: u64, // raw device pointer
        rows: i64,
        cols: i64,
        ld: i64,
    ) -> Result<Self, GpuError> {
        let mut desc = MaybeUninit::uninit();
        unsafe {
            csp::cusparseCreateDnMat(
                desc.as_mut_ptr(),
                rows,
                cols,
                ld,
                values as *mut core::ffi::c_void,
                cudaDataType::CUDA_R_32F,
                csp::cusparseOrder_t::CUSPARSE_ORDER_COL,
            )
            .result()
            .map_err(|e| GpuError::CuSparseError(format!("cusparseCreateDnMat: {e:?}")))?;
            Ok(Self {
                raw: desc.assume_init(),
            })
        }
    }

    /// Access the raw descriptor for FFI calls.
    pub fn raw(&self) -> csp::cusparseDnMatDescr_t {
        self.raw
    }
}

impl Drop for DnMatDescr {
    fn drop(&mut self) {
        unsafe {
            let _ = csp::cusparseDestroyDnMat(self.raw);
        }
    }
}

/// A strided view into a column-major dense matrix buffer.
///
/// Describes a sub-region of `buf` interpreted as a `(rows × cols)` column-major
/// matrix with leading dimension `ld`, starting at `buf[offset_elems]`.
///
/// Used by the strided cuSPARSE SpMM wrappers ([`spmm_csr_view`] /
/// [`spmm_csr_transpose_view`]) to let SpMM write/read a sub-region of a
/// larger global buffer without per-shard scatter/gather kernels. `ld` must
/// be >= `rows` for the contiguous case; for sub-regions, `ld` is the leading
/// dimension of the full enclosing matrix (e.g. `n_obs` when writing a
/// `shard_rows × k` view into a `n_obs × k` buffer at row offset `global_row`,
/// in which case `offset_elems = global_row`).
#[derive(Copy, Clone)]
pub struct DnMatView<'a> {
    pub buf: &'a CudaSlice<f32>,
    pub offset_elems: usize,
    pub rows: i64,
    pub cols: i64,
    pub ld: i64,
}

/// Mutable counterpart to [`DnMatView`].
pub struct DnMatViewMut<'a> {
    pub buf: &'a mut CudaSlice<f32>,
    pub offset_elems: usize,
    pub rows: i64,
    pub cols: i64,
    pub ld: i64,
}

impl<'a> DnMatView<'a> {
    /// Construct a contiguous view (offset = 0, ld = rows).
    pub fn contiguous(buf: &'a CudaSlice<f32>, rows: i64, cols: i64) -> Self {
        Self {
            buf,
            offset_elems: 0,
            rows,
            cols,
            ld: rows,
        }
    }
}

impl<'a> DnMatViewMut<'a> {
    /// Construct a contiguous mutable view (offset = 0, ld = rows).
    pub fn contiguous(buf: &'a mut CudaSlice<f32>, rows: i64, cols: i64) -> Self {
        Self {
            buf,
            offset_elems: 0,
            rows,
            cols,
            ld: rows,
        }
    }
}

/// Pool of reusable cuSPARSE SpMM workspace buffers.
///
/// `spmm_csr` allocates a fresh `CudaSlice<u8>` workspace inside every call.
/// In iterative GPU PCA (randomized power iteration) that means ~30 × N_shards
/// × 2 allocations per run. The pool replaces that with one grow-only
/// `CudaSlice<u8>` slot:
///
/// 1. First call queries `cusparseSpMM_bufferSize` for the required size,
///    allocates the slot, and runs SpMM.
/// 2. Subsequent calls with `buf_size <= slot capacity` reuse the slot
///    without allocating.
/// 3. Calls with a larger `buf_size` grow the slot to `next_power_of_two`
///    and record the realloc.
///
/// The atomic counters `alloc_count` / `reuse_count` expose pool behaviour
/// for tests (see [`Self::metrics`]) — they're not load-bearing for
/// correctness.
///
/// Thread safety: the pool itself is `!Sync` because a SpMM call needs
/// exclusive access to the workspace buffer for the duration of the kernel.
/// Run one pool per stream / device context.
pub struct CuSparseWorkspacePool {
    slot: Option<CudaSlice<u8>>,
    capacity_bytes: usize,
    alloc_count: AtomicU64,
    reuse_count: AtomicU64,
}

/// Snapshot of pool usage counters. Returned by [`CuSparseWorkspacePool::metrics`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CuSparsePoolMetrics {
    /// Number of slot allocations (initial + growths).
    pub alloc_count: u64,
    /// Number of calls served from the existing slot without allocation.
    pub reuse_count: u64,
    /// Current slot capacity in bytes (0 if never allocated).
    pub current_capacity_bytes: usize,
}

impl Default for CuSparseWorkspacePool {
    fn default() -> Self {
        Self::new()
    }
}

impl CuSparseWorkspacePool {
    /// Construct an empty pool. The first SpMM call lazily allocates the slot.
    pub fn new() -> Self {
        Self {
            slot: None,
            capacity_bytes: 0,
            alloc_count: AtomicU64::new(0),
            reuse_count: AtomicU64::new(0),
        }
    }

    /// Snapshot the pool's alloc/reuse counters and current capacity.
    pub fn metrics(&self) -> CuSparsePoolMetrics {
        CuSparsePoolMetrics {
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            reuse_count: self.reuse_count.load(Ordering::Relaxed),
            current_capacity_bytes: self.capacity_bytes,
        }
    }

    /// Grow the slot if `bytes > current capacity`. No-op when zero-bytes are
    /// requested (cuSPARSE returns `buf_size = 0` for some trivial SpMM
    /// shapes) — counted as a reuse so the metrics make sense.
    fn ensure_capacity(&mut self, dev: &GpuDevice, bytes: usize) -> Result<(), GpuError> {
        if bytes == 0 {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        if bytes <= self.capacity_bytes {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        // Bump to next power-of-two so repeated small growths amortise.
        let new_cap = bytes.next_power_of_two().max(self.capacity_bytes * 2);
        self.slot = Some(dev.alloc_zeros::<u8>(new_cap)?);
        self.capacity_bytes = new_cap;
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Lend the slot to a closure that performs one SpMM call. The closure
    /// receives a `*mut c_void` workspace pointer (or null when `bytes == 0`)
    /// — pass it straight to `cusparseSpMM`. The pool grows on first call /
    /// on size increase.
    fn with_workspace<R>(
        &mut self,
        dev: &GpuDevice,
        stream: &CudaStream,
        bytes: usize,
        f: impl FnOnce(*mut core::ffi::c_void) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.ensure_capacity(dev, bytes)?;
        if let Some(slot) = self.slot.as_mut() {
            let (ptr, _guard) = slot.device_ptr_mut(stream);
            // `_guard` holds the borrow on `slot` until end of scope, which
            // includes the closure call. Required so concurrent reads/writes
            // through cudarc's tracking don't reuse the slot mid-kernel.
            f(ptr as *mut core::ffi::c_void)
        } else {
            f(std::ptr::null_mut())
        }
    }
}

/// Compute sparse × dense matrix multiply: `C = α·A·B + β·C`.
///
/// - `A` is a GPU-resident CSR matrix (m × k) via `CusparseSpMatDescr`
/// - `B` is a dense **column-major** matrix (k × n) on GPU
/// - `C` is a dense **column-major** matrix (m × n) on GPU, overwritten
///
/// Uses cuSPARSE generic SpMM API with:
/// - `CUSPARSE_OPERATION_NON_TRANSPOSE` for both A and B
/// - `CUSPARSE_ORDER_COL` (cuSPARSE preference for performance)
/// - `CUDA_R_32F` compute type
/// - `CUSPARSE_SPMM_ALG_DEFAULT` (cuSPARSE auto-tunes)
///
/// The workspace buffer is allocated per-call. For repeated SpMM calls with
/// the same sparsity pattern (e.g., power iteration in PCA), consider
/// caching the workspace externally.
#[allow(clippy::too_many_arguments)]
pub fn spmm_csr(
    handle: &CusparseHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    a: &CusparseSpMatDescr,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), GpuError> {
    spmm_impl(
        handle,
        stream,
        dev,
        csp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        a,
        b,
        c,
        m,
        k,
        n,
        alpha,
        beta,
    )
}

/// SpMM with transpose: `C = α·A^T·B + β·C`.
///
/// - `A` is a GPU-resident CSR matrix (m × k), transposed to (k × m)
/// - `B` is a dense **column-major** matrix (m × n) on GPU
/// - `C` is a dense **column-major** matrix (k × n) on GPU, overwritten
///
/// Same algorithm selection as [`spmm_csr`].
#[allow(clippy::too_many_arguments)]
pub fn spmm_csr_transpose(
    handle: &CusparseHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    a: &CusparseSpMatDescr,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), GpuError> {
    spmm_impl(
        handle,
        stream,
        dev,
        csp::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        a,
        b,
        c,
        m,
        k,
        n,
        alpha,
        beta,
    )
}

/// Strided SpMM: `C = α·op(A)·B + β·C` where `B` and `C` are strided views
/// into possibly-larger column-major buffers.
///
/// This is the unified entry point that the contiguous helpers ([`spmm_csr`],
/// [`spmm_csr_transpose`]) and the strided helpers ([`spmm_csr_view`],
/// [`spmm_csr_transpose_view`]) all delegate to. `pool` is `Some` when the
/// caller wants the workspace reused across calls (PCA power iteration);
/// `None` falls back to per-call `dev.alloc_zeros`.
#[allow(clippy::too_many_arguments)]
fn spmm_impl_view(
    handle: &CusparseHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    pool: Option<&mut CuSparseWorkspacePool>,
    op_a: csp::cusparseOperation_t,
    a: &CusparseSpMatDescr,
    b: DnMatView<'_>,
    c: DnMatViewMut<'_>,
    alpha: f32,
    beta: f32,
) -> Result<(), GpuError> {
    // Bind cuSPARSE handle to our CUDA stream
    handle.set_stream(stream)?;

    // Resolve the strided base pointers via cudarc slice views. The guards
    // borrow from the underlying slices for the full SpMM call so cudarc's
    // synchronization tracking is honoured.
    let b_view = b.buf.slice(b.offset_elems..);
    let (b_ptr, _guard_b) = b_view.device_ptr(stream);
    let mut c_view = c.buf.slice_mut(c.offset_elems..);
    let (c_ptr, _guard_c) = c_view.device_ptr_mut(stream);

    // Create dense matrix descriptors with the caller-provided leading dims.
    let dn_b = DnMatDescr::new(b_ptr, b.rows, b.cols, b.ld)?;
    let dn_c = DnMatDescr::new(c_ptr, c.rows, c.cols, c.ld)?;

    let alpha_ptr = &alpha as *const f32 as *const core::ffi::c_void;
    let beta_ptr = &beta as *const f32 as *const core::ffi::c_void;
    let alg = csp::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT;

    // Query workspace buffer size
    let mut buf_size: usize = 0;
    unsafe {
        csp::cusparseSpMM_bufferSize(
            handle.raw(),
            op_a,
            csp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha_ptr,
            a.raw(),
            dn_b.raw(),
            beta_ptr,
            dn_c.raw(),
            cudaDataType::CUDA_R_32F,
            alg,
            &mut buf_size as *mut usize,
        )
        .result()
        .map_err(|e| GpuError::CuSparseError(format!("cusparseSpMM_bufferSize: {e:?}")))?;
    }

    // Branch on whether the caller supplied a workspace pool.
    let run_spmm = |workspace_ptr: *mut core::ffi::c_void| -> Result<(), GpuError> {
        unsafe {
            csp::cusparseSpMM(
                handle.raw(),
                op_a,
                csp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha_ptr,
                a.raw(),
                dn_b.raw(),
                beta_ptr,
                dn_c.raw(),
                cudaDataType::CUDA_R_32F,
                alg,
                workspace_ptr,
            )
            .result()
            .map_err(|e| GpuError::CuSparseError(format!("cusparseSpMM: {e:?}")))?;
        }
        Ok(())
    };

    match pool {
        Some(p) => p.with_workspace(dev, stream, buf_size, run_spmm)?,
        None => {
            // Legacy per-call allocation path.
            let workspace = if buf_size > 0 {
                Some(dev.alloc_zeros::<u8>(buf_size)?)
            } else {
                None
            };
            let workspace_ptr = match &workspace {
                Some(ws) => {
                    let (ptr, _guard) = ws.device_ptr(stream);
                    ptr as *mut core::ffi::c_void
                }
                None => std::ptr::null_mut(),
            };
            run_spmm(workspace_ptr)?;
        }
    }

    Ok(())
}

/// Convert `(m, k, n, op_a)` plus contiguous buffers into the equivalent
/// strided-view dimensions for the legacy `spmm_csr` / `spmm_csr_transpose`
/// entry points. Centralises the dimension table shared by both wrappers.
fn contiguous_view_dims(
    op_a: csp::cusparseOperation_t,
    m: usize,
    k: usize,
    n: usize,
) -> (i64, i64, i64, i64) {
    match op_a {
        csp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE => {
            (k as i64, n as i64, m as i64, n as i64)
        }
        // TRANSPOSE or CONJUGATE_TRANSPOSE
        _ => (m as i64, n as i64, k as i64, n as i64),
    }
}

/// Legacy contiguous-buffer SpMM, retained for callers that don't yet thread
/// a `CuSparseWorkspacePool` through. Delegates to [`spmm_impl_view`].
#[allow(clippy::too_many_arguments)]
fn spmm_impl(
    handle: &CusparseHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    op_a: csp::cusparseOperation_t,
    a: &CusparseSpMatDescr,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), GpuError> {
    let (b_rows, b_cols, c_rows, c_cols) = contiguous_view_dims(op_a, m, k, n);
    let b_view = DnMatView {
        buf: b,
        offset_elems: 0,
        rows: b_rows,
        cols: b_cols,
        ld: b_rows,
    };
    let c_view = DnMatViewMut {
        buf: c,
        offset_elems: 0,
        rows: c_rows,
        cols: c_cols,
        ld: c_rows,
    };
    spmm_impl_view(
        handle, stream, dev, None, op_a, a, b_view, c_view, alpha, beta,
    )
}

/// Strided SpMM: `C = α·A·B + β·C` where `B` / `C` are views into larger
/// column-major buffers.
///
/// Same algorithm and tuning as [`spmm_csr`]. Used by the GPU PCA loop to
/// write a `(shard_rows × k)` SpMM result directly into a `(n_obs × k)`
/// global buffer at row offset `global_row`, avoiding a per-shard scatter.
///
/// `b.ld` and `c.ld` must each be `>= rows`. `pool` is `Some` to reuse
/// workspace across calls (recommended in iterative loops); `None` allocates
/// per call.
#[allow(clippy::too_many_arguments)]
pub fn spmm_csr_view(
    handle: &CusparseHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    pool: Option<&mut CuSparseWorkspacePool>,
    a: &CusparseSpMatDescr,
    b: DnMatView<'_>,
    c: DnMatViewMut<'_>,
    alpha: f32,
    beta: f32,
) -> Result<(), GpuError> {
    spmm_impl_view(
        handle,
        stream,
        dev,
        pool,
        csp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        a,
        b,
        c,
        alpha,
        beta,
    )
}

/// Strided transposed SpMM: `C = α·Aᵀ·B + β·C` where `B` / `C` are views
/// into larger column-major buffers.
///
/// Same algorithm and tuning as [`spmm_csr_transpose`]. Used by the GPU PCA
/// loop to read a `(shard_rows × k)` view from a `(n_obs × k)` buffer at
/// row offset `global_row`, avoiding a per-shard gather.
#[allow(clippy::too_many_arguments)]
pub fn spmm_csr_transpose_view(
    handle: &CusparseHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    pool: Option<&mut CuSparseWorkspacePool>,
    a: &CusparseSpMatDescr,
    b: DnMatView<'_>,
    c: DnMatViewMut<'_>,
    alpha: f32,
    beta: f32,
) -> Result<(), GpuError> {
    spmm_impl_view(
        handle,
        stream,
        dev,
        pool,
        csp::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        a,
        b,
        c,
        alpha,
        beta,
    )
}

/// Raw device pointers for cupy `__cuda_array_interface__` interop.
///
/// All pointers are `u64` (CUDA device pointers). The consumer is responsible
/// for interpreting them with the correct dtype and shape.
#[derive(Debug, Clone, Copy)]
pub struct GpuCsrPointers {
    /// Device pointer to `i64` indptr array (n_rows + 1 elements).
    pub indptr_ptr: u64,
    /// Device pointer to `i32` indices array (nnz elements).
    pub indices_ptr: u64,
    /// Device pointer to `f32` data array (nnz elements).
    pub data_ptr: u64,
    /// Number of non-zero elements.
    pub nnz: usize,
    /// Matrix shape (n_rows, n_cols).
    pub shape: (usize, usize),
}

impl GpuCsr {
    /// Create a cuSPARSE CSR sparse matrix descriptor.
    ///
    /// The descriptor references the **column indices**, **values**, and a
    /// **downcast i32 copy of indptr** that is owned by the returned
    /// descriptor. The original i64 indptr is NOT referenced. This `GpuCsr`
    /// **must outlive** the returned descriptor for `indices` / `data`; the
    /// downcast indptr lives as long as the descriptor itself.
    ///
    /// Type mapping:
    /// - indptr: `CUSPARSE_INDEX_32I` (downcast from i64; per-shard nnz is
    ///   always < 2^31 for single-cell data)
    /// - indices: `CUSPARSE_INDEX_32I` (i32)
    /// - data: `CUDA_R_32F` (f32)
    /// - index base: zero-based
    ///
    /// # Why the downcast?
    ///
    /// Some cuSPARSE releases (observed on CUDA 12.1.2.141 — driver 535 /
    /// cuda-version 12.2 conda env) return
    /// `CUSPARSE_STATUS_NOT_SUPPORTED` when `csrRowOffsetsType` and
    /// `csrColIndType` disagree (mixed 64I/32I). Matching them on 32I is
    /// safe because per-shard nnz fits in i32 for all realistic workloads
    /// (census_1m ≈ 10⁸ nnz/shard at most, well below 2^31 ≈ 2.1 × 10⁹).
    pub fn to_cusparse_csr(
        &self,
        dev: &GpuDevice,
        stream: &CudaStream,
    ) -> Result<CusparseSpMatDescr, GpuError> {
        let (n_rows, n_cols) = self.shape;
        let nnz = self.indices.len();

        // Downcast indptr i64 → i32 on-device (cheap — O(n_rows+1), one pass).
        // Stored inside the descriptor so cuSPARSE's captured pointer remains
        // valid for the descriptor's lifetime.
        let i32_indptr = cast_i64_to_i32_gpu(dev, stream, &self.indptr)?;

        // Capture the raw device pointers inside an inner scope so the
        // SyncOnDrop guards (which borrow from `i32_indptr` / `self.*`)
        // are released before we move `i32_indptr` into the descriptor.
        let desc_raw = {
            let (indptr_ptr, _guard_indptr) = i32_indptr.device_ptr(stream);
            let (indices_ptr, _guard_indices) = self.indices.device_ptr(stream);
            let (data_ptr, _guard_data) = self.data.device_ptr(stream);
            let mut desc = MaybeUninit::uninit();
            unsafe {
                csp::cusparseCreateCsr(
                    desc.as_mut_ptr(),
                    n_rows as i64,
                    n_cols as i64,
                    nnz as i64,
                    indptr_ptr as *mut core::ffi::c_void,
                    indices_ptr as *mut core::ffi::c_void,
                    data_ptr as *mut core::ffi::c_void,
                    cusparseIndexType_t::CUSPARSE_INDEX_32I,
                    cusparseIndexType_t::CUSPARSE_INDEX_32I,
                    cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                    cudaDataType::CUDA_R_32F,
                )
                .result()
                .map_err(|e| GpuError::CuSparseError(format!("cusparseCreateCsr: {e:?}")))?;
                desc.assume_init()
            }
        };

        Ok(CusparseSpMatDescr {
            raw: desc_raw,
            _i32_indptr: Some(i32_indptr),
        })
    }

    /// Expose raw device pointers for cupy `__cuda_array_interface__` interop.
    ///
    /// Returns device pointer addresses (u64) that can be passed to Python
    /// via PyO3 for zero-copy access from cupy/torch.
    ///
    /// # Lifetime
    ///
    /// The returned pointer values (`u64`) are valid only while `self` is alive.
    /// The caller must ensure the `GpuCsr` outlives any use of these pointers.
    /// The `SyncOnDrop` guards from `device_ptr()` are dropped at the end of
    /// this method, recording read events for synchronization tracking.
    pub fn device_pointers(&self, stream: &CudaStream) -> GpuCsrPointers {
        let (indptr_ptr, _g1) = self.indptr.device_ptr(stream);
        let (indices_ptr, _g2) = self.indices.device_ptr(stream);
        let (data_ptr, _g3) = self.data.device_ptr(stream);

        GpuCsrPointers {
            indptr_ptr,
            indices_ptr,
            data_ptr,
            nnz: self.indices.len(),
            shape: self.shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::shard_decode::decode_shard_gpu;
    use crate::test_utils::build_test_shard;
    use scx_codec::{CodecId, ValueEncoding};

    /// The cuSPARSE ABI probe must return a bool without panicking, even when
    /// no libcusparse is installed (the "no GPU runtime" case on CPU-only CI
    /// runners). Regardless of GPU availability, calling the probe twice
    /// returns the same cached value — `OnceLock` semantics.
    #[test]
    fn test_cusparse_modern_abi_probe_does_not_panic() {
        let first = cusparse_modern_abi_available();
        let second = cusparse_modern_abi_available();
        assert_eq!(
            first, second,
            "probe is OnceLock-cached; the two reads must agree"
        );
        // Don't assert the value itself — it depends on whether libcusparse
        // is installed and whether it predates cuSPARSE 12.5 on this host.
        // The contract is "doesn't panic and is idempotent".
    }

    /// Build a small test shard for cuSPARSE tests.
    fn build_small_shard() -> Vec<u8> {
        let indptr = vec![0u64, 3, 5, 5, 8, 12];
        let indices = vec![
            0u32, 10, 50, // row 0
            5, 15, // row 1
            // row 2 empty
            1, 2, 3, // row 3
            20, 40, 60, 80, // row 4
        ];
        let values_u16: Vec<u16> = (1..=12).collect();
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            100,
        )
    }

    #[test]
    fn test_cusparse_descriptor() {
        let dev = require_gpu!();
        let shard_bytes = build_small_shard();
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();

        // Create cuSPARSE handle
        let _handle = CusparseHandle::new().unwrap();

        // Create CSR descriptor — this is zero-copy, just wraps the pointers
        let desc = gpu_csr.to_cusparse_csr(&dev, dev.stream()).unwrap();

        // The descriptor should be non-null
        assert!(
            !desc.raw().is_null(),
            "cuSPARSE descriptor should be non-null"
        );

        // Verify shape
        assert_eq!(gpu_csr.shape, (5, 100));
        assert_eq!(gpu_csr.indices.len(), 12);
        assert_eq!(gpu_csr.indptr.len(), 6); // n_rows + 1

        // Descriptor is dropped here, calling cusparseDestroySpMat
    }

    #[test]
    fn test_device_pointers() {
        let dev = require_gpu!();
        let shard_bytes = build_small_shard();
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();

        let ptrs = gpu_csr.device_pointers(dev.stream());

        // Device pointers should be non-zero (allocated on GPU)
        assert_ne!(
            ptrs.indptr_ptr, 0,
            "indptr device pointer should be non-zero"
        );
        assert_ne!(
            ptrs.indices_ptr, 0,
            "indices device pointer should be non-zero"
        );
        assert_ne!(ptrs.data_ptr, 0, "data device pointer should be non-zero");

        // All pointers should be distinct
        assert_ne!(ptrs.indptr_ptr, ptrs.indices_ptr);
        assert_ne!(ptrs.indices_ptr, ptrs.data_ptr);
        assert_ne!(ptrs.indptr_ptr, ptrs.data_ptr);

        // Metadata should match
        assert_eq!(ptrs.nnz, 12);
        assert_eq!(ptrs.shape, (5, 100));
    }

    /// CPU reference SpMM: C = A * B where A is CSR, B is column-major dense.
    ///
    /// A: (m × k), B: (k × n) col-major, C: (m × n) col-major.
    fn cpu_spmm(
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        b: &[f32], // col-major (k × n)
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let _ = k; // k is implicit in CSR structure
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for nz in start..end {
                let col_a = indices[nz] as usize;
                let val_a = data[nz];
                for j in 0..n {
                    // B is col-major: B[col_a, j] = b[j * k + col_a]
                    // C is col-major: C[row, j]   = c[j * m + row]
                    c[j * m + row] += val_a * b[j * k + col_a];
                }
            }
        }
        c
    }

    /// CPU reference SpMM transpose: C = A^T * B
    ///
    /// A: (m × k), A^T: (k × m), B: (m × n) col-major, C: (k × n) col-major.
    fn cpu_spmm_transpose(
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        b: &[f32], // col-major (m × n)
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let _ = k; // k is implicit
        let mut c = vec![0.0f32; k * n];
        // A^T[col_a, row] = A[row, col_a]
        for row in 0..m {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for nz in start..end {
                let col_a = indices[nz] as usize;
                let val_a = data[nz];
                for j in 0..n {
                    // B is col-major: B[row, j] = b[j * m + row]
                    // C is col-major: C[col_a, j] = c[j * k + col_a]
                    c[j * k + col_a] += val_a * b[j * m + row];
                }
            }
        }
        c
    }

    /// Build a simple CSR for SpMM tests (no SCX codec encoding needed).
    fn build_simple_gpu_csr(
        dev: &GpuDevice,
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> GpuCsr {
        let d_indptr = dev.htod_copy(indptr).unwrap();
        let d_indices = dev.htod_copy(indices).unwrap();
        let d_data = dev.htod_copy(data).unwrap();
        GpuCsr {
            indptr: d_indptr,
            indices: d_indices,
            data: d_data,
            shape: (n_rows, n_cols),
        }
    }

    #[test]
    fn test_spmm_csr() {
        let dev = require_gpu!();

        // A: 4×3 sparse CSR
        //   row 0: [(0, 1.0), (2, 3.0)]
        //   row 1: [(1, 2.0)]
        //   row 2: []
        //   row 3: [(0, 4.0), (1, 5.0), (2, 6.0)]
        let m = 4;
        let k = 3;
        let n = 2;
        let indptr: Vec<i64> = vec![0, 2, 3, 3, 6];
        let indices: Vec<i32> = vec![0, 2, 1, 0, 1, 2];
        let data: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];

        // B: 3×2 col-major:  [[1, 4],
        //                      [2, 5],
        //                      [3, 6]]
        // col-major: [1,2,3, 4,5,6]
        let b_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gpu_csr = build_simple_gpu_csr(&dev, &indptr, &indices, &data, m, k);
        let handle = CusparseHandle::new().unwrap();
        let a_desc = gpu_csr.to_cusparse_csr(&dev, dev.stream()).unwrap();

        let d_b = dev.htod_copy(&b_host).unwrap();
        let mut d_c = dev.alloc_zeros::<f32>(m * n).unwrap();

        spmm_csr(
            &handle,
            dev.stream(),
            &dev,
            &a_desc,
            &d_b,
            &mut d_c,
            m,
            k,
            n,
            1.0,
            0.0,
        )
        .unwrap();

        let c_gpu = dev.dtoh_copy(&d_c).unwrap();
        let c_cpu = cpu_spmm(&indptr, &indices, &data, &b_host, m, k, n);

        assert_eq!(c_gpu.len(), c_cpu.len());
        for i in 0..c_gpu.len() {
            assert!(
                (c_gpu[i] - c_cpu[i]).abs() < 1e-5,
                "SpMM mismatch at index {i}: GPU={}, CPU={}",
                c_gpu[i],
                c_cpu[i]
            );
        }
    }

    #[test]
    fn test_spmm_csr_transpose() {
        let dev = require_gpu!();

        // A: 4×3 sparse CSR (same as above)
        // A^T: 3×4, B: 4×2 col-major, C: 3×2 col-major
        let m = 4;
        let k = 3;
        let n = 2;
        let indptr: Vec<i64> = vec![0, 2, 3, 3, 6];
        let indices: Vec<i32> = vec![0, 2, 1, 0, 1, 2];
        let data: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];

        // B: 4×2 col-major
        let b_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let gpu_csr = build_simple_gpu_csr(&dev, &indptr, &indices, &data, m, k);
        let handle = CusparseHandle::new().unwrap();
        let a_desc = gpu_csr.to_cusparse_csr(&dev, dev.stream()).unwrap();

        let d_b = dev.htod_copy(&b_host).unwrap();
        let mut d_c = dev.alloc_zeros::<f32>(k * n).unwrap();

        super::spmm_csr_transpose(
            &handle,
            dev.stream(),
            &dev,
            &a_desc,
            &d_b,
            &mut d_c,
            m,
            k,
            n,
            1.0,
            0.0,
        )
        .unwrap();

        let c_gpu = dev.dtoh_copy(&d_c).unwrap();
        let c_cpu = cpu_spmm_transpose(&indptr, &indices, &data, &b_host, m, k, n);

        assert_eq!(c_gpu.len(), c_cpu.len());
        for i in 0..c_gpu.len() {
            assert!(
                (c_gpu[i] - c_cpu[i]).abs() < 1e-5,
                "SpMM transpose mismatch at index {i}: GPU={}, CPU={}",
                c_gpu[i],
                c_cpu[i]
            );
        }
    }

    /// Strided SpMM view parity vs the contiguous `spmm_csr`.
    ///
    /// Writes a `(m × n)` result into an oversized `(m + 5) × n` buffer at
    /// row offset 3 with `ld = m + 5`, then verifies the populated rows
    /// match the contiguous-buffer result bit-for-bit and that the
    /// untouched rows remain zero.
    #[test]
    fn test_spmm_csr_view_matches_contiguous() {
        let dev = require_gpu!();
        let m = 4;
        let k = 3;
        let n = 2;
        let indptr: Vec<i64> = vec![0, 2, 3, 3, 6];
        let indices: Vec<i32> = vec![0, 2, 1, 0, 1, 2];
        let data: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
        let b_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gpu_csr = build_simple_gpu_csr(&dev, &indptr, &indices, &data, m, k);
        let handle = CusparseHandle::new().unwrap();
        let a_desc = gpu_csr.to_cusparse_csr(&dev, dev.stream()).unwrap();

        // Reference: contiguous SpMM.
        let d_b = dev.htod_copy(&b_host).unwrap();
        let mut d_c_contig = dev.alloc_zeros::<f32>(m * n).unwrap();
        spmm_csr(
            &handle,
            dev.stream(),
            &dev,
            &a_desc,
            &d_b,
            &mut d_c_contig,
            m,
            k,
            n,
            1.0,
            0.0,
        )
        .unwrap();
        let c_ref = dev.dtoh_copy(&d_c_contig).unwrap();

        // Strided: write into a `(ld × n)` buffer at row offset 3 with ld = m + 5.
        let ld = m + 5;
        let row_off = 3usize;
        let mut d_c_strided = dev.alloc_zeros::<f32>(ld * n).unwrap();

        let b_view = DnMatView::contiguous(&d_b, k as i64, n as i64);
        let c_view = DnMatViewMut {
            buf: &mut d_c_strided,
            offset_elems: row_off,
            rows: m as i64,
            cols: n as i64,
            ld: ld as i64,
        };
        spmm_csr_view(
            &handle,
            dev.stream(),
            &dev,
            None,
            &a_desc,
            b_view,
            c_view,
            1.0,
            0.0,
        )
        .unwrap();
        let c_strided = dev.dtoh_copy(&d_c_strided).unwrap();

        // Verify the populated sub-region matches the contiguous result.
        for col in 0..n {
            for row in 0..m {
                let strided_idx = col * ld + row_off + row;
                let contig_idx = col * m + row;
                assert!(
                    (c_strided[strided_idx] - c_ref[contig_idx]).abs() < 1e-5,
                    "strided[{strided_idx}]={} != contiguous[{contig_idx}]={}",
                    c_strided[strided_idx],
                    c_ref[contig_idx]
                );
            }
        }

        // Verify untouched rows remain zero — the strided write must not
        // bleed into rows outside `[row_off, row_off + m)`.
        for col in 0..n {
            for row in 0..ld {
                if row >= row_off && row < row_off + m {
                    continue;
                }
                let idx = col * ld + row;
                assert_eq!(
                    c_strided[idx], 0.0,
                    "strided write leaked into untouched row {row} (col {col})"
                );
            }
        }
    }

    /// Same as the forward parity test but for the transpose path. Reads a
    /// strided `(m × n)` view from a `(ld × n)` source buffer.
    #[test]
    fn test_spmm_csr_transpose_view_matches_contiguous() {
        let dev = require_gpu!();
        let m = 4;
        let k = 3;
        let n = 2;
        let indptr: Vec<i64> = vec![0, 2, 3, 3, 6];
        let indices: Vec<i32> = vec![0, 2, 1, 0, 1, 2];
        let data: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
        let b_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let gpu_csr = build_simple_gpu_csr(&dev, &indptr, &indices, &data, m, k);
        let handle = CusparseHandle::new().unwrap();
        let a_desc = gpu_csr.to_cusparse_csr(&dev, dev.stream()).unwrap();

        // Reference: contiguous transposed SpMM.
        let d_b_contig = dev.htod_copy(&b_host).unwrap();
        let mut d_c_contig = dev.alloc_zeros::<f32>(k * n).unwrap();
        super::spmm_csr_transpose(
            &handle,
            dev.stream(),
            &dev,
            &a_desc,
            &d_b_contig,
            &mut d_c_contig,
            m,
            k,
            n,
            1.0,
            0.0,
        )
        .unwrap();
        let c_ref = dev.dtoh_copy(&d_c_contig).unwrap();

        // Strided: read B from a (ld × n) buffer at row offset 3 with ld = m + 5.
        let ld = m + 5;
        let row_off = 3usize;
        let mut b_strided_host = vec![0.0f32; ld * n];
        for col in 0..n {
            for row in 0..m {
                b_strided_host[col * ld + row_off + row] = b_host[col * m + row];
            }
        }
        let d_b_strided = dev.htod_copy(&b_strided_host).unwrap();
        let mut d_c_view = dev.alloc_zeros::<f32>(k * n).unwrap();

        let b_view = DnMatView {
            buf: &d_b_strided,
            offset_elems: row_off,
            rows: m as i64,
            cols: n as i64,
            ld: ld as i64,
        };
        let c_view = DnMatViewMut::contiguous(&mut d_c_view, k as i64, n as i64);
        spmm_csr_transpose_view(
            &handle,
            dev.stream(),
            &dev,
            None,
            &a_desc,
            b_view,
            c_view,
            1.0,
            0.0,
        )
        .unwrap();
        let c_view_host = dev.dtoh_copy(&d_c_view).unwrap();

        assert_eq!(c_view_host.len(), c_ref.len());
        for i in 0..c_view_host.len() {
            assert!(
                (c_view_host[i] - c_ref[i]).abs() < 1e-5,
                "transpose view mismatch at {i}: view={}, contiguous={}",
                c_view_host[i],
                c_ref[i]
            );
        }
    }

    /// The pool must reuse its single grow-only slot across SpMM calls of the
    /// same shape.
    #[test]
    fn test_workspace_pool_reuses_across_calls() {
        let dev = require_gpu!();
        let m = 4;
        let k = 3;
        let n = 2;
        let indptr: Vec<i64> = vec![0, 2, 3, 3, 6];
        let indices: Vec<i32> = vec![0, 2, 1, 0, 1, 2];
        let data: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
        let b_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gpu_csr = build_simple_gpu_csr(&dev, &indptr, &indices, &data, m, k);
        let handle = CusparseHandle::new().unwrap();
        let a_desc = gpu_csr.to_cusparse_csr(&dev, dev.stream()).unwrap();
        let d_b = dev.htod_copy(&b_host).unwrap();
        let mut d_c = dev.alloc_zeros::<f32>(m * n).unwrap();
        let mut pool = CuSparseWorkspacePool::new();

        for _ in 0..10 {
            let b_view = DnMatView::contiguous(&d_b, k as i64, n as i64);
            let c_view = DnMatViewMut::contiguous(&mut d_c, m as i64, n as i64);
            spmm_csr_view(
                &handle,
                dev.stream(),
                &dev,
                Some(&mut pool),
                &a_desc,
                b_view,
                c_view,
                1.0,
                0.0,
            )
            .unwrap();
        }

        let metrics = pool.metrics();
        // First call allocates (or pool.with_workspace counts a 0-byte reuse
        // when cuSPARSE returns buf_size = 0). Either way:
        // - alloc_count ≤ 1 (a single grow event for the largest workspace).
        // - alloc_count + reuse_count == 10 (one increment per call).
        assert!(
            metrics.alloc_count <= 1,
            "expected at most 1 allocation; got {} (capacity {} bytes)",
            metrics.alloc_count,
            metrics.current_capacity_bytes
        );
        assert_eq!(
            metrics.alloc_count + metrics.reuse_count,
            10,
            "pool counters should sum to call count"
        );
    }

    /// The pool must grow exactly once when a larger shape arrives.
    #[test]
    fn test_workspace_pool_grows_on_bigger_shape() {
        let dev = require_gpu!();
        // Small CSR: 4×3.
        let m_small = 4;
        let k_small = 3;
        let n_small = 2;
        let indptr_s: Vec<i64> = vec![0, 2, 3, 3, 6];
        let indices_s: Vec<i32> = vec![0, 2, 1, 0, 1, 2];
        let data_s: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
        let b_small: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Larger CSR: 200×100 dense-ish (synthetic, just to force a bigger
        // workspace). One nonzero per row.
        let m_big = 200;
        let k_big = 100;
        let n_big = 32;
        let mut indptr_b: Vec<i64> = Vec::with_capacity(m_big + 1);
        let mut indices_b: Vec<i32> = Vec::with_capacity(m_big);
        let mut data_b: Vec<f32> = Vec::with_capacity(m_big);
        indptr_b.push(0);
        for row in 0..m_big {
            indices_b.push((row % k_big) as i32);
            data_b.push(1.0);
            indptr_b.push(indices_b.len() as i64);
        }
        let b_big: Vec<f32> = vec![1.0f32; k_big * n_big];

        let handle = CusparseHandle::new().unwrap();
        let mut pool = CuSparseWorkspacePool::new();

        // Small call first.
        {
            let csr_s =
                build_simple_gpu_csr(&dev, &indptr_s, &indices_s, &data_s, m_small, k_small);
            let a_s = csr_s.to_cusparse_csr(&dev, dev.stream()).unwrap();
            let d_b = dev.htod_copy(&b_small).unwrap();
            let mut d_c = dev.alloc_zeros::<f32>(m_small * n_small).unwrap();
            let b_view = DnMatView::contiguous(&d_b, k_small as i64, n_small as i64);
            let c_view = DnMatViewMut::contiguous(&mut d_c, m_small as i64, n_small as i64);
            spmm_csr_view(
                &handle,
                dev.stream(),
                &dev,
                Some(&mut pool),
                &a_s,
                b_view,
                c_view,
                1.0,
                0.0,
            )
            .unwrap();
        }
        let after_small = pool.metrics();

        // Big call — should grow the slot.
        {
            let csr_b = build_simple_gpu_csr(&dev, &indptr_b, &indices_b, &data_b, m_big, k_big);
            let a_b = csr_b.to_cusparse_csr(&dev, dev.stream()).unwrap();
            let d_b = dev.htod_copy(&b_big).unwrap();
            let mut d_c = dev.alloc_zeros::<f32>(m_big * n_big).unwrap();
            let b_view = DnMatView::contiguous(&d_b, k_big as i64, n_big as i64);
            let c_view = DnMatViewMut::contiguous(&mut d_c, m_big as i64, n_big as i64);
            spmm_csr_view(
                &handle,
                dev.stream(),
                &dev,
                Some(&mut pool),
                &a_b,
                b_view,
                c_view,
                1.0,
                0.0,
            )
            .unwrap();
        }
        let after_big = pool.metrics();

        // If the small call required workspace (buf_size > 0), it allocated
        // exactly once. The big call either reused (small workspace was
        // already big enough — alloc_count unchanged) or grew (alloc_count
        // bumped by 1). Cap at 2 total allocations regardless of cuSPARSE's
        // internal sizing decisions.
        assert!(
            after_big.alloc_count <= 2,
            "expected ≤ 2 grow events, got {}",
            after_big.alloc_count
        );
        assert!(
            after_big.alloc_count + after_big.reuse_count == 2,
            "pool counters should sum to 2 calls (got alloc={} reuse={})",
            after_big.alloc_count,
            after_big.reuse_count
        );
        // If the small call did force an allocation, the big call should not
        // have shrunk the slot — capacity is monotonic non-decreasing.
        assert!(
            after_big.current_capacity_bytes >= after_small.current_capacity_bytes,
            "pool capacity must be monotonic non-decreasing"
        );
    }

    #[test]
    fn test_spmm_alpha_beta() {
        let dev = require_gpu!();

        // Test alpha/beta scaling: C = 2.0 * A * B + 0.5 * C
        let m = 2;
        let k = 2;
        let n = 1;
        let indptr: Vec<i64> = vec![0, 1, 2];
        let indices: Vec<i32> = vec![0, 1];
        let data: Vec<f32> = vec![3.0, 4.0];

        // B = [1.0, 2.0] col-major (2×1)
        let b_host: Vec<f32> = vec![1.0, 2.0];
        // C_init = [10.0, 20.0] col-major (2×1)
        let c_init: Vec<f32> = vec![10.0, 20.0];

        let gpu_csr = build_simple_gpu_csr(&dev, &indptr, &indices, &data, m, k);
        let handle = CusparseHandle::new().unwrap();
        let a_desc = gpu_csr.to_cusparse_csr(&dev, dev.stream()).unwrap();

        let d_b = dev.htod_copy(&b_host).unwrap();
        let mut d_c = dev.htod_copy(&c_init).unwrap();

        spmm_csr(
            &handle,
            dev.stream(),
            &dev,
            &a_desc,
            &d_b,
            &mut d_c,
            m,
            k,
            n,
            2.0,
            0.5,
        )
        .unwrap();

        let c_gpu = dev.dtoh_copy(&d_c).unwrap();
        // A*B = [3*1, 4*2] = [3, 8]
        // C = 2.0*[3, 8] + 0.5*[10, 20] = [6, 16] + [5, 10] = [11, 26]
        assert!((c_gpu[0] - 11.0).abs() < 1e-5);
        assert!((c_gpu[1] - 26.0).abs() < 1e-5);
    }
}
