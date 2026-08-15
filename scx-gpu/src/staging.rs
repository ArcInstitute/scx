//! Shard staging primitives for asynchronous host→device CSR upload.
//!
//! `PinnedCsrSlot` and `GpuCsrSlot` are paired grow-only buffers used by the
//! `GpuShardSource` adapters (see [`crate::gpu_shard_source`]) to amortise
//! per-shard allocation and unblock truly-async H→D copies.
//!
//! ## Pinned host staging
//!
//! `PinnedCsrSlot` holds three host buffers (`indptr`, `indices`, `data`).
//! On a CUDA host they are allocated as page-locked / write-combined memory
//! via `cuMemHostAlloc`, which is the only host-side layout that lets
//! `memcpy_htod_async` actually run asynchronously on a dedicated copy
//! stream (pageable host buffers force the driver to stage through an
//! internal pinned bounce buffer and block the calling thread). If the
//! pinned allocation fails (out of pinned memory, or driver does not
//! support page-locking the requested layout), the slot transparently
//! falls back to pageable `Vec<T>` — the upload still works, just without
//! the async win.
//!
//! ## Reusable device CSR slot
//!
//! `GpuCsrSlot` owns three grow-only `CudaSlice` buffers plus an optional
//! cached `CusparseSpMatDescr`. The slot is filled via
//! [`GpuCsrSlot::upload_from_pinned`] and yields exact-sized `CudaView`s
//! through [`GpuCsrSlot::view`]. Repeated uploads of the same logical
//! shape (e.g. PCA power iteration over the same shard) reuse the
//! descriptor; growing the underlying buffers invalidates the cache.
//!
//! Both slot types are `!Sync` — they're designed for single-stream
//! single-thread consumers (the main thread of `RawGpuShardSource`, one pool
//! per device per pipeline).

use std::sync::Arc;

use cudarc::driver::safe::{
    CudaContext, CudaSlice, CudaStream, CudaView, DevicePtr, PinnedHostSlice,
};

use scx_sparse::{ScxCsc, ScxCsr};

use crate::cusparse::CusparseSpMatDescr;
use crate::device::GpuDevice;
use crate::error::GpuError;

/// Borrowed CSR view backed by a `GpuCsrSlot`.
///
/// All fields are exact-sized — `indptr.len() == n_rows + 1`,
/// `indices.len() == data.len() == nnz`. Callers can pass `&indptr`,
/// `&indices`, `&data` directly to kernel launches and cuSPARSE descriptor
/// builders.
pub struct GpuCsrShardView<'a> {
    pub indptr: CudaView<'a, i64>,
    pub indices: CudaView<'a, i32>,
    pub data: CudaView<'a, f32>,
    pub shape: (usize, usize),
}

impl<'a> GpuCsrShardView<'a> {
    /// nnz from the view's data length (matches the indptr's last entry).
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

// --------------------------------------------------------------------------
// PinnedCsrSlot
// --------------------------------------------------------------------------

/// Pinned host buffer holding one component of a CSR shard.
///
/// `Pinned` is the win path — page-locked memory allocated via cudarc's
/// `alloc_pinned`. `Pageable` is a transparent fallback when the pinned
/// allocation fails. Both expose the same `as_slice` / `as_mut_slice`
/// interface; downstream `memcpy_htod` calls accept either through the
/// `HostSlice<T>` trait.
enum HostBuf<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits> {
    Pinned(PinnedHostSlice<T>),
    Pageable(Vec<T>),
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone + Default> HostBuf<T> {
    /// Try pinned; fall back to pageable on `cuMemHostAlloc` failure.
    fn new(ctx: &Arc<CudaContext>, capacity: usize) -> Self {
        // SAFETY: alloc_pinned is `unsafe` because the returned memory is
        // uninitialised. We only ever write to it before reading (via
        // memcpy_htod which copies from a pre-filled slice, or via
        // explicit `fill_from`). No reads of uninitialised memory happen
        // through this slot.
        match unsafe { ctx.alloc_pinned::<T>(capacity) } {
            Ok(slot) => HostBuf::Pinned(slot),
            Err(_) => HostBuf::Pageable(vec![T::default(); capacity]),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            HostBuf::Pinned(p) => p.len(),
            HostBuf::Pageable(v) => v.len(),
        }
    }

    /// Copy `src` into the leading `src.len()` elements of the buffer.
    /// Caller must have already called `ensure_capacity(ctx, src.len())`.
    fn fill_from(&mut self, src: &[T]) -> Result<(), GpuError> {
        match self {
            HostBuf::Pinned(p) => {
                let slice = p
                    .as_mut_slice()
                    .map_err(|e| GpuError::CudaError(format!("pinned as_mut_slice: {e}")))?;
                debug_assert!(slice.len() >= src.len());
                slice[..src.len()].clone_from_slice(src);
                Ok(())
            }
            HostBuf::Pageable(v) => {
                debug_assert!(v.len() >= src.len());
                v[..src.len()].clone_from_slice(src);
                Ok(())
            }
        }
    }
}

/// Three pinned-or-pageable host buffers staging a CSR shard's
/// `(indptr, indices, data)` arrays.
///
/// Grow-only: `ensure_capacity` may reallocate any buffer that needs to be
/// larger, but never shrinks. The buffers' `len()` always reflects the
/// allocated capacity, not the currently-populated shard size — the active
/// shard's dimensions are tracked separately on the consumer side.
pub struct PinnedCsrSlot {
    ctx: Arc<CudaContext>,
    indptr: HostBuf<i64>,
    indices: HostBuf<i32>,
    data: HostBuf<f32>,
    /// True if at least one buffer is pinned. Surfaced for tests / metrics
    /// so callers can detect the pageable fallback.
    is_pinned: bool,
}

impl PinnedCsrSlot {
    /// Construct a slot with initial capacities (in elements). On a host
    /// without pinned-memory support the slot transparently falls back to
    /// pageable Vecs.
    pub fn new(ctx: &Arc<CudaContext>, indptr_cap: usize, nnz_cap: usize) -> Self {
        let indptr = HostBuf::<i64>::new(ctx, indptr_cap.max(1));
        let indices = HostBuf::<i32>::new(ctx, nnz_cap.max(1));
        let data = HostBuf::<f32>::new(ctx, nnz_cap.max(1));
        let is_pinned = matches!(indptr, HostBuf::Pinned(_))
            && matches!(indices, HostBuf::Pinned(_))
            && matches!(data, HostBuf::Pinned(_));
        Self {
            ctx: ctx.clone(),
            indptr,
            indices,
            data,
            is_pinned,
        }
    }

    /// Whether all three buffers are pinned (true) or any fell back to
    /// pageable (false). Useful for tests and per-shard metrics.
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Grow buffers if needed. No-op when current capacity already
    /// suffices.
    ///
    /// `is_pinned` is monotone-downward only: a pinned → pageable grow
    /// flips it to `false`, but a subsequent pageable → pinned grow does
    /// not flip it back. The flag is metric-only and not load-bearing for
    /// correctness — H→D code paths dispatch per-buffer on the actual
    /// `HostBuf` variant.
    pub fn ensure_capacity(&mut self, indptr_len: usize, nnz: usize) {
        if self.indptr.capacity() < indptr_len {
            let new_cap = indptr_len.next_power_of_two();
            let new_buf = HostBuf::<i64>::new(&self.ctx, new_cap);
            if matches!(self.indptr, HostBuf::Pinned(_)) && !matches!(new_buf, HostBuf::Pinned(_)) {
                self.is_pinned = false;
            }
            self.indptr = new_buf;
        }
        if self.indices.capacity() < nnz {
            let new_cap = nnz.next_power_of_two();
            let new_buf = HostBuf::<i32>::new(&self.ctx, new_cap);
            if matches!(self.indices, HostBuf::Pinned(_)) && !matches!(new_buf, HostBuf::Pinned(_))
            {
                self.is_pinned = false;
            }
            self.indices = new_buf;
        }
        if self.data.capacity() < nnz {
            let new_cap = nnz.next_power_of_two();
            let new_buf = HostBuf::<f32>::new(&self.ctx, new_cap);
            if matches!(self.data, HostBuf::Pinned(_)) && !matches!(new_buf, HostBuf::Pinned(_)) {
                self.is_pinned = false;
            }
            self.data = new_buf;
        }
    }

    /// Stage a `ScxCsr` shard into the host slot. Grows the slot if needed.
    pub fn stage(&mut self, csr: &ScxCsr) -> Result<(), GpuError> {
        self.ensure_capacity(csr.indptr.len(), csr.data.len());
        self.indptr.fill_from(&csr.indptr)?;
        self.indices.fill_from(&csr.indices)?;
        self.data.fill_from(&csr.data)?;
        Ok(())
    }

    /// Issue async `memcpy_htod` of the staged shard onto `stream` into the
    /// device slot. The device slot is grown to fit first.
    ///
    /// `n_rows` and `nnz` describe the live shard sizes; the buffers are
    /// expected to already contain a valid shard from `stage()`.
    pub fn upload_to(
        &self,
        stream: &Arc<CudaStream>,
        dst: &mut GpuCsrSlot,
        n_rows: usize,
        nnz: usize,
        n_cols: usize,
    ) -> Result<(), GpuError> {
        dst.ensure_capacity(stream, n_rows + 1, nnz)?;

        macro_rules! upload {
            ($src:expr, $dst:expr, $len:expr) => {{
                match $src {
                    HostBuf::Pinned(p) => {
                        // Memcpy a sub-prefix using `slice` view of the
                        // device buffer, but cudarc's `memcpy_htod` copies
                        // src.len() elements; we need to wrap the pinned
                        // slot in a slice of the live length.
                        let host_slice = p
                            .as_slice()
                            .map_err(|e| GpuError::CudaError(format!("pinned slice: {e}")))?;
                        let mut dst_view = $dst.slice_mut(..$len);
                        stream
                            .memcpy_htod(&host_slice[..$len], &mut dst_view)
                            .map_err(|e| GpuError::CudaError(format!("htod async: {e}")))?;
                    }
                    HostBuf::Pageable(v) => {
                        let mut dst_view = $dst.slice_mut(..$len);
                        stream
                            .memcpy_htod(&v[..$len], &mut dst_view)
                            .map_err(|e| GpuError::CudaError(format!("htod sync: {e}")))?;
                    }
                }
            }};
        }
        upload!(&self.indptr, &mut dst.indptr, n_rows + 1);
        upload!(&self.indices, &mut dst.indices, nnz);
        upload!(&self.data, &mut dst.data, nnz);

        dst.shape = (n_rows, n_cols);
        dst.nnz = nnz;
        // Shape-change invalidation is required for correctness:
        // `cusparseCreateCsr` captures `(rows, cols, nnz)` at descriptor
        // build time (see [`build_sp_descr_from_slot`]), so a descriptor
        // built for one shape would silently produce wrong results if
        // reused after a shape change.
        dst.maybe_invalidate_descr_on_shape_change(n_rows, n_cols, nnz);
        Ok(())
    }
}

// --------------------------------------------------------------------------
// PinnedCscSlot
// --------------------------------------------------------------------------

/// Three pinned-or-pageable host buffers staging a CSC shard's
/// `(col_indptr, row_indices, data)` arrays.
///
/// Column-major counterpart to [`PinnedCsrSlot`]. Used by the pipelined
/// [`crate::gpu_csc_shard_source::RawGpuCscShardSource`] adapter to
/// overlap shard decode + H→D copy with the previous shard's GPU
/// compute, in the same shape as the CSR `RawGpuShardSource`.
///
/// Grow-only: `ensure_capacity` may reallocate any buffer that needs to
/// be larger, but never shrinks. Pinned alloc failure transparently
/// falls back to pageable for each buffer independently.
pub struct PinnedCscSlot {
    ctx: Arc<CudaContext>,
    col_indptr: HostBuf<i64>,
    row_indices: HostBuf<i32>,
    data: HostBuf<f32>,
    /// True if all three buffers are currently pinned. Monotone-downward
    /// on `ensure_capacity` if any grow falls back to pageable; not
    /// load-bearing for correctness — H→D dispatches on the actual
    /// `HostBuf` variant per buffer.
    is_pinned: bool,
}

impl PinnedCscSlot {
    /// Construct a slot with initial capacities (in elements).
    pub fn new(ctx: &Arc<CudaContext>, col_indptr_cap: usize, nnz_cap: usize) -> Self {
        let col_indptr = HostBuf::<i64>::new(ctx, col_indptr_cap.max(1));
        let row_indices = HostBuf::<i32>::new(ctx, nnz_cap.max(1));
        let data = HostBuf::<f32>::new(ctx, nnz_cap.max(1));
        let is_pinned = matches!(col_indptr, HostBuf::Pinned(_))
            && matches!(row_indices, HostBuf::Pinned(_))
            && matches!(data, HostBuf::Pinned(_));
        Self {
            ctx: ctx.clone(),
            col_indptr,
            row_indices,
            data,
            is_pinned,
        }
    }

    /// Whether all three buffers are pinned (true) or any fell back to
    /// pageable (false). Useful for tests and per-shard metrics.
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Grow buffers if needed. No-op when current capacity suffices.
    pub fn ensure_capacity(&mut self, col_indptr_len: usize, nnz: usize) {
        if self.col_indptr.capacity() < col_indptr_len {
            let new_cap = col_indptr_len.next_power_of_two();
            let new_buf = HostBuf::<i64>::new(&self.ctx, new_cap);
            if matches!(self.col_indptr, HostBuf::Pinned(_))
                && !matches!(new_buf, HostBuf::Pinned(_))
            {
                self.is_pinned = false;
            }
            self.col_indptr = new_buf;
        }
        if self.row_indices.capacity() < nnz {
            let new_cap = nnz.next_power_of_two();
            let new_buf = HostBuf::<i32>::new(&self.ctx, new_cap);
            if matches!(self.row_indices, HostBuf::Pinned(_))
                && !matches!(new_buf, HostBuf::Pinned(_))
            {
                self.is_pinned = false;
            }
            self.row_indices = new_buf;
        }
        if self.data.capacity() < nnz {
            let new_cap = nnz.next_power_of_two();
            let new_buf = HostBuf::<f32>::new(&self.ctx, new_cap);
            if matches!(self.data, HostBuf::Pinned(_)) && !matches!(new_buf, HostBuf::Pinned(_)) {
                self.is_pinned = false;
            }
            self.data = new_buf;
        }
    }

    /// Stage an [`ScxCsc`] shard into the host slot. Grows the slot if
    /// needed.
    pub fn stage(&mut self, csc: &ScxCsc) -> Result<(), GpuError> {
        self.ensure_capacity(csc.indptr.len(), csc.data.len());
        self.col_indptr.fill_from(&csc.indptr)?;
        self.row_indices.fill_from(&csc.indices)?;
        self.data.fill_from(&csc.data)?;
        Ok(())
    }

    /// Issue async H→D from the pinned slot into the supplied device
    /// buffers on `stream`.
    ///
    /// The destination `CudaSlice`s must already have capacity ≥
    /// `col_indptr_len` (for `dst_col_indptr`) / `nnz` (for
    /// `dst_row_indices` and `dst_data`). Callers grow them via the
    /// outer source's own capacity tracking before invoking this method.
    pub fn upload_to(
        &self,
        stream: &Arc<CudaStream>,
        dst_col_indptr: &mut CudaSlice<i64>,
        dst_row_indices: &mut CudaSlice<i32>,
        dst_data: &mut CudaSlice<f32>,
        col_indptr_len: usize,
        nnz: usize,
    ) -> Result<(), GpuError> {
        macro_rules! upload {
            ($src:expr, $dst:expr, $len:expr) => {{
                match $src {
                    HostBuf::Pinned(p) => {
                        let host_slice = p
                            .as_slice()
                            .map_err(|e| GpuError::CudaError(format!("pinned slice: {e}")))?;
                        let mut dst_view = $dst.slice_mut(..$len);
                        stream
                            .memcpy_htod(&host_slice[..$len], &mut dst_view)
                            .map_err(|e| GpuError::CudaError(format!("htod async: {e}")))?;
                    }
                    HostBuf::Pageable(v) => {
                        let mut dst_view = $dst.slice_mut(..$len);
                        stream
                            .memcpy_htod(&v[..$len], &mut dst_view)
                            .map_err(|e| GpuError::CudaError(format!("htod sync: {e}")))?;
                    }
                }
            }};
        }
        upload!(&self.col_indptr, dst_col_indptr, col_indptr_len);
        upload!(&self.row_indices, dst_row_indices, nnz);
        upload!(&self.data, dst_data, nnz);
        Ok(())
    }
}

// --------------------------------------------------------------------------
// GpuCsrSlot
// --------------------------------------------------------------------------

/// Grow-only device CSR buffer.
///
/// Owned device storage that survives across shard uploads — sized once at
/// catalog time (via `max_shard_rows()` / `nnz`) and reused. Repeated
/// uploads of the same shape reuse the cached `CusparseSpMatDescr` (item
/// G3.3).
///
/// `shape` and `nnz` reflect the **currently-staged** shard, not the
/// buffer capacity. The slot's three `CudaSlice` buffers may be larger.
pub struct GpuCsrSlot {
    indptr: CudaSlice<i64>,
    indices: CudaSlice<i32>,
    data: CudaSlice<f32>,
    shape: (usize, usize),
    nnz: usize,
    /// Cached cuSPARSE descriptor over the slot's current buffers. Keyed
    /// by `(shape, nnz, buffer_ptrs)` — invalidated on grow or shape
    /// change. `OnceCell`-style: built lazily by [`Self::cached_sp_descr`].
    cached_desc: Option<CachedDesc>,
}

/// Cached descriptor with the signature that produced it.
struct CachedDesc {
    descr: CusparseSpMatDescr,
    sig: DescrSig,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct DescrSig {
    n_rows: usize,
    n_cols: usize,
    nnz: usize,
    indptr_ptr: u64,
    indices_ptr: u64,
    data_ptr: u64,
}

impl GpuCsrSlot {
    /// Construct a slot pre-sized for the largest expected shard.
    ///
    /// `max_indptr_len` is `max_shard_rows + 1`; `max_nnz` is the largest
    /// expected nnz per shard. Both can be conservative — buffers grow on
    /// demand via [`Self::ensure_capacity`].
    pub fn new(dev: &GpuDevice, max_indptr_len: usize, max_nnz: usize) -> Result<Self, GpuError> {
        let indptr = dev.alloc_zeros::<i64>(max_indptr_len.max(1))?;
        let indices = dev.alloc_zeros::<i32>(max_nnz.max(1))?;
        let data = dev.alloc_zeros::<f32>(max_nnz.max(1))?;
        Ok(Self {
            indptr,
            indices,
            data,
            shape: (0, 0),
            nnz: 0,
            cached_desc: None,
        })
    }

    /// Deep-copy the **live shard** into a fresh, exactly-sized slot.
    ///
    /// Unlike [`Self::new`] + [`Self::ensure_capacity`] (which round buffer
    /// capacity up to the next power of two, so a 24 M-nnz shard reserves
    /// 33.5 M), this allocates exactly `n_rows + 1` / `nnz` / `nnz` elements
    /// and `memcpy_dtod`s the staged prefix across. Nothing round-trips to the
    /// host.
    ///
    /// Used by [`ResidentGpuCsrSource`](crate::ResidentGpuCsrSource) to retain
    /// every shard on the device: a streaming source stages each shard into one
    /// reusable slot and overwrites it on the next iteration, so retaining a
    /// shard means copying it out. Exact sizing matters there — the retained
    /// set is the whole matrix, and a power-of-two round-up on each shard would
    /// inflate resident VRAM by up to 2× for no benefit.
    ///
    /// The cached cuSPARSE descriptor is **not** copied: it is keyed by device
    /// pointer, and the clone's buffers are new allocations. The clone rebuilds
    /// it lazily on first use, like any freshly-grown slot.
    pub fn clone_exact(&self, dev: &GpuDevice) -> Result<Self, GpuError> {
        let indptr_len = self.shape.0 + 1;
        let mut indptr = dev.alloc_zeros::<i64>(indptr_len.max(1))?;
        let mut indices = dev.alloc_zeros::<i32>(self.nnz.max(1))?;
        let mut data = dev.alloc_zeros::<f32>(self.nnz.max(1))?;

        let stream = dev.stream();
        {
            let src = self.indptr.slice(..indptr_len);
            let mut dst = indptr.slice_mut(..indptr_len);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| GpuError::CudaError(format!("dtod indptr (clone_exact): {e}")))?;
        }
        if self.nnz > 0 {
            let src = self.indices.slice(..self.nnz);
            let mut dst = indices.slice_mut(..self.nnz);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| GpuError::CudaError(format!("dtod indices (clone_exact): {e}")))?;
            let src = self.data.slice(..self.nnz);
            let mut dst = data.slice_mut(..self.nnz);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| GpuError::CudaError(format!("dtod data (clone_exact): {e}")))?;
        }

        Ok(Self {
            indptr,
            indices,
            data,
            shape: self.shape,
            nnz: self.nnz,
            cached_desc: None,
        })
    }

    /// Total device bytes this slot's three buffers occupy (capacity, not the
    /// live shard). Used by the resident-CSR VRAM budget.
    pub fn device_bytes(&self) -> u64 {
        (self.indptr.len() as u64) * 8
            + (self.indices.len() as u64) * 4
            + (self.data.len() as u64) * 4
    }

    /// Capacity (in elements) of the three buffers.
    pub fn capacity(&self) -> (usize, usize, usize) {
        (self.indptr.len(), self.indices.len(), self.data.len())
    }

    /// Currently-staged shard's shape and nnz.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Grow buffers if needed. No-op when current capacity suffices.
    ///
    /// Re-allocation invalidates the cached cuSPARSE descriptor (the new
    /// buffers have different device pointers).
    pub fn ensure_capacity(
        &mut self,
        stream: &Arc<CudaStream>,
        indptr_len: usize,
        nnz: usize,
    ) -> Result<(), GpuError> {
        if self.indptr.len() < indptr_len {
            let new_cap = indptr_len.next_power_of_two();
            self.indptr = stream
                .alloc_zeros::<i64>(new_cap)
                .map_err(|e| GpuError::OutOfMemory(format!("grow indptr: {e}")))?;
            self.cached_desc = None;
        }
        if self.indices.len() < nnz {
            let new_cap = nnz.next_power_of_two();
            self.indices = stream
                .alloc_zeros::<i32>(new_cap)
                .map_err(|e| GpuError::OutOfMemory(format!("grow indices: {e}")))?;
            self.cached_desc = None;
        }
        if self.data.len() < nnz {
            let new_cap = nnz.next_power_of_two();
            self.data = stream
                .alloc_zeros::<f32>(new_cap)
                .map_err(|e| GpuError::OutOfMemory(format!("grow data: {e}")))?;
            self.cached_desc = None;
        }
        Ok(())
    }

    /// Borrow exact-sized views over the live shard. Each view has
    /// `len()` equal to the staged shard's `n_rows + 1` / `nnz` /
    /// `nnz` — not the underlying buffer capacity.
    pub fn view(&self) -> GpuCsrShardView<'_> {
        GpuCsrShardView {
            indptr: self.indptr.slice(..self.shape.0 + 1),
            indices: self.indices.slice(..self.nnz),
            data: self.data.slice(..self.nnz),
            shape: self.shape,
        }
    }

    /// Mutable view of the `data` buffer for in-place transforms. The
    /// view is exact-sized at the live shard's nnz. The `indptr` view
    /// rides along unchanged so the consumer (a normalize / log1p kernel)
    /// can walk rows.
    pub fn data_mut(&mut self) -> cudarc::driver::safe::CudaViewMut<'_, f32> {
        self.data.slice_mut(..self.nnz)
    }

    pub fn indptr_view(&self) -> CudaView<'_, i64> {
        self.indptr.slice(..self.shape.0 + 1)
    }

    pub fn indices_view(&self) -> CudaView<'_, i32> {
        self.indices.slice(..self.nnz)
    }

    /// Borrow `indptr` immutably and `data` mutably at the same time.
    ///
    /// Required by in-place preprocessing kernels that read row offsets
    /// from `indptr` while writing nonzeros in `data`. The two borrows
    /// touch distinct fields, so Rust's field-level borrow tracking
    /// permits the split; expose it as a method so callers don't have to
    /// reach into the slot's private fields.
    pub fn split_indptr_data_mut(
        &mut self,
    ) -> (
        CudaView<'_, i64>,
        cudarc::driver::safe::CudaViewMut<'_, f32>,
    ) {
        let n_rows_plus_1 = self.shape.0 + 1;
        let nnz = self.nnz;
        (
            self.indptr.slice(..n_rows_plus_1),
            self.data.slice_mut(..nnz),
        )
    }

    /// Get-or-create the cuSPARSE CSR descriptor for the current shard.
    ///
    /// Returns the cached descriptor when the slot's pointers and shape
    /// haven't changed since the last call. Otherwise rebuilds it via
    /// [`GpuCsr::to_cusparse_csr`]. The cache is invalidated on
    /// buffer-grow (see [`Self::ensure_capacity`]) and on shape change
    /// (tracked via the signature).
    pub fn cached_sp_descr(
        &mut self,
        dev: &GpuDevice,
        stream: &CudaStream,
    ) -> Result<&CusparseSpMatDescr, GpuError> {
        let sig = self.current_descr_sig(stream);
        let cache_valid = self
            .cached_desc
            .as_ref()
            .map(|c| c.sig == sig)
            .unwrap_or(false);
        if !cache_valid {
            // Build the cuSPARSE descriptor directly against the slot's
            // exact-sized views — `GpuCsr::to_cusparse_csr` would require
            // an owned `GpuCsr`, which would defeat the slot's buffer
            // reuse. See [`build_sp_descr_from_slot`].
            let descr = build_sp_descr_from_slot(dev, stream, self)?;
            self.cached_desc = Some(CachedDesc { descr, sig });
        }
        // SAFETY: `cached_desc` is `Some` here.
        Ok(&self.cached_desc.as_ref().unwrap().descr)
    }

    fn current_descr_sig(&self, stream: &CudaStream) -> DescrSig {
        let (indptr_ptr, _g1) = self.indptr.device_ptr(stream);
        let (indices_ptr, _g2) = self.indices.device_ptr(stream);
        let (data_ptr, _g3) = self.data.device_ptr(stream);
        DescrSig {
            n_rows: self.shape.0,
            n_cols: self.shape.1,
            nnz: self.nnz,
            indptr_ptr,
            indices_ptr,
            data_ptr,
        }
    }

    fn maybe_invalidate_descr_on_shape_change(&mut self, n_rows: usize, n_cols: usize, nnz: usize) {
        if let Some(c) = &self.cached_desc {
            if c.sig.n_rows != n_rows || c.sig.n_cols != n_cols || c.sig.nnz != nnz {
                self.cached_desc = None;
            }
        }
    }

    /// True if a cached cuSPARSE descriptor is currently held. Exposed for
    /// tests to assert "descriptor was reused across N calls".
    pub fn has_cached_descr(&self) -> bool {
        self.cached_desc.is_some()
    }
}

// --------------------------------------------------------------------------
// InMemoryCsrShardSource
// --------------------------------------------------------------------------

/// Adapter exposing a borrowed [`ScxCsr`] as a single-shard
/// [`scx_format_io::ShardSource`].
///
/// Lets the in-memory-CSR arm of the unified GPU DE entry points
/// (`pdex_ref_gpu` / `wilcoxon_rank_sum_gpu` with `GpuDeShardInput::Csr`) feed
/// the refactored chunked driver,
/// which consumes any `&dyn ShardSource + Sync` through
/// [`crate::gpu_shard_source::RawGpuShardSource`]. The driver's per-shard
/// device-resident gene-major scatter replaced the per-chunk host
/// materialise + dense-upload path, which has since been deleted.
///
/// `read_shard(0)` returns `csr.clone()` — `ScxCsr` owns its buffers, so
/// cloning is an `Arc`-free vector copy. The cost is paid once at the
/// start of the call and amortised across every chunk; the alternative
/// (avoiding the clone with a `Cow`-style trait change) would force a
/// signature break on `ShardSource` for a one-shot adapter.
pub struct InMemoryCsrShardSource<'a> {
    csr: &'a ScxCsr,
}

impl<'a> InMemoryCsrShardSource<'a> {
    pub fn new(csr: &'a ScxCsr) -> Self {
        Self { csr }
    }
}

impl<'a> scx_format_io::ShardSource for InMemoryCsrShardSource<'a> {
    fn n_shards(&self) -> usize {
        1
    }
    fn n_obs(&self) -> usize {
        self.csr.shape.0
    }
    fn n_vars(&self) -> usize {
        self.csr.shape.1
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
        if shard_idx != 0 {
            return Err(scx_format_io::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: 1,
            });
        }
        Ok(self.csr.clone())
    }
    fn max_shard_rows(&self) -> scx_format_io::Result<usize> {
        Ok(self.csr.shape.0)
    }
}

/// Build a cuSPARSE CSR descriptor pointing at the slot's live buffers.
///
/// Functionally equivalent to [`GpuCsr::to_cusparse_csr`] but works on
/// the slot's existing buffers (no `try_clone`, no temporary `GpuCsr`).
/// The descriptor downcasts indptr i64 → i32 on-device and stores the
/// downcast buffer inside the descriptor so cuSPARSE's captured pointer
/// remains valid for the descriptor's lifetime.
///
/// The `stream` argument is plumbed through to the cast kernel so the
/// downcast `indptr` is produced on the same stream that the cuSPARSE
/// call will use — without it, the cast would run on `dev.stream()` and
/// the descriptor would capture a pointer whose contents are not visible
/// on the caller's stream until an implicit synchronization.
fn build_sp_descr_from_slot(
    dev: &GpuDevice,
    stream: &CudaStream,
    slot: &GpuCsrSlot,
) -> Result<CusparseSpMatDescr, GpuError> {
    use crate::cast_gpu::cast_i64_to_i32_gpu_view;
    let indptr_view = slot.indptr.slice(..slot.shape.0 + 1);
    let i32_indptr = cast_i64_to_i32_gpu_view(dev, stream, &indptr_view)?;

    use cudarc::cusparse::sys::{
        self as csp, cudaDataType, cusparseIndexBase_t, cusparseIndexType_t,
    };
    use cudarc::driver::safe::DevicePtr;
    use std::mem::MaybeUninit;

    let desc_raw = {
        let (indptr_ptr, _g1) = i32_indptr.device_ptr(stream);
        let indices_view = slot.indices.slice(..slot.nnz);
        let data_view = slot.data.slice(..slot.nnz);
        let (indices_ptr, _g2) = indices_view.device_ptr(stream);
        let (data_ptr, _g3) = data_view.device_ptr(stream);

        let mut desc = MaybeUninit::uninit();
        unsafe {
            csp::cusparseCreateCsr(
                desc.as_mut_ptr(),
                slot.shape.0 as i64,
                slot.shape.1 as i64,
                slot.nnz as i64,
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

    // Hand ownership of the i32 indptr into the descriptor via a tiny
    // adapter — see CusparseSpMatDescr in cusparse.rs for the rationale.
    Ok(CusparseSpMatDescr::from_raw_with_i32_indptr(
        desc_raw, i32_indptr,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_pinned_slot_basic_stage_and_grow() {
        let dev = require_gpu!();
        let ctx = dev.context();
        let mut slot = PinnedCsrSlot::new(ctx, 4, 4);
        let (cap_i, cap_x, cap_d) = (
            match &slot.indptr {
                HostBuf::Pinned(p) => p.len(),
                HostBuf::Pageable(v) => v.len(),
            },
            match &slot.indices {
                HostBuf::Pinned(p) => p.len(),
                HostBuf::Pageable(v) => v.len(),
            },
            match &slot.data {
                HostBuf::Pinned(p) => p.len(),
                HostBuf::Pageable(v) => v.len(),
            },
        );
        assert!(cap_i >= 4 && cap_x >= 4 && cap_d >= 4);

        // Stage a tiny shard.
        let csr = ScxCsr::new_unchecked(
            (3, 5),
            vec![0i64, 2, 3, 5],
            vec![0i32, 2, 1, 0, 4],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
        );
        slot.stage(&csr).unwrap();

        // Grow.
        slot.ensure_capacity(200, 1000);
        // After grow, capacity must accommodate; old buffers may have been
        // freed and replaced. Buffer pinned-ness may degrade if pinned
        // alloc fails — we don't assert anything about is_pinned here
        // because the test environment may or may not have pinned memory.
        let big = ScxCsr::new_unchecked(
            (150, 5),
            (0..151).map(|i| i as i64).collect::<Vec<_>>(),
            (0..150).map(|_| 0i32).collect::<Vec<_>>(),
            (0..150).map(|i| i as f32).collect::<Vec<_>>(),
        );
        slot.stage(&big).unwrap();
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_csr_slot_upload_and_view_sizes() {
        let dev = require_gpu!();
        let ctx = dev.context();

        let mut host = PinnedCsrSlot::new(ctx, 16, 64);
        let mut slot = GpuCsrSlot::new(&dev, 16, 64).unwrap();
        let csr = ScxCsr::new_unchecked(
            (4, 8),
            vec![0i64, 2, 3, 3, 6],
            vec![0i32, 4, 2, 1, 4, 7],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        host.stage(&csr).unwrap();
        host.upload_to(dev.stream(), &mut slot, 4, 6, 8).unwrap();
        dev.synchronize().unwrap();

        assert_eq!(slot.shape(), (4, 8));
        assert_eq!(slot.nnz(), 6);

        let v = slot.view();
        assert_eq!(v.indptr.len(), 5);
        assert_eq!(v.indices.len(), 6);
        assert_eq!(v.data.len(), 6);

        // Round-trip the data back to host and check equality with the
        // staged values.
        let mut dst = vec![0i64; 5];
        dev.stream().memcpy_dtoh(&v.indptr, &mut dst).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(dst, vec![0, 2, 3, 3, 6]);

        let mut data_dst = vec![0.0f32; 6];
        dev.stream().memcpy_dtoh(&v.data, &mut data_dst).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(data_dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_csr_slot_descr_cache_hit_same_shape() {
        let dev = require_gpu!();
        let ctx = dev.context();
        let mut host = PinnedCsrSlot::new(ctx, 16, 64);
        let mut slot = GpuCsrSlot::new(&dev, 16, 64).unwrap();
        let csr = ScxCsr::new_unchecked(
            (4, 8),
            vec![0i64, 2, 3, 3, 6],
            vec![0i32, 4, 2, 1, 4, 7],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        host.stage(&csr).unwrap();
        host.upload_to(dev.stream(), &mut slot, 4, 6, 8).unwrap();

        assert!(!slot.has_cached_descr());
        let p1 = slot.cached_sp_descr(&dev, dev.stream()).unwrap().raw();
        assert!(slot.has_cached_descr());
        // Second call without reupload — descr pointer should match.
        let p2 = slot.cached_sp_descr(&dev, dev.stream()).unwrap().raw();
        assert_eq!(
            p1, p2,
            "descriptor must be reused across calls with same shape"
        );
    }

    /// Descriptor cache must invalidate when only `nnz` changes (same
    /// n_rows / n_cols). The existing `*_on_shape_change` test varies
    /// `n_rows`; this isolates the nnz path.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_csr_slot_descr_cache_invalidates_on_nnz_change() {
        let dev = require_gpu!();
        let ctx = dev.context();
        let mut host = PinnedCsrSlot::new(ctx, 16, 64);
        let mut slot = GpuCsrSlot::new(&dev, 16, 64).unwrap();

        // Shape A: 3 rows × 5 cols, nnz = 5.
        let csr_a = ScxCsr::new_unchecked(
            (3, 5),
            vec![0i64, 2, 3, 5],
            vec![0i32, 2, 1, 0, 4],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
        );
        host.stage(&csr_a).unwrap();
        host.upload_to(dev.stream(), &mut slot, 3, 5, 5).unwrap();
        let _p1 = slot.cached_sp_descr(&dev, dev.stream()).unwrap().raw();
        assert!(slot.has_cached_descr());

        // Shape B: SAME n_rows / n_cols, DIFFERENT nnz (4 instead of 5).
        let csr_b = ScxCsr::new_unchecked(
            (3, 5),
            vec![0i64, 2, 3, 4],
            vec![0i32, 2, 1, 0],
            vec![10.0f32, 11.0, 12.0, 13.0],
        );
        host.stage(&csr_b).unwrap();
        host.upload_to(dev.stream(), &mut slot, 3, 4, 5).unwrap();
        assert!(
            !slot.has_cached_descr(),
            "nnz change must invalidate descr cache even when n_rows/n_cols are unchanged"
        );
        let _p2 = slot.cached_sp_descr(&dev, dev.stream()).unwrap().raw();
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_csr_slot_descr_cache_invalidates_on_shape_change() {
        let dev = require_gpu!();
        let ctx = dev.context();
        let mut host = PinnedCsrSlot::new(ctx, 16, 64);
        let mut slot = GpuCsrSlot::new(&dev, 16, 64).unwrap();
        let csr_a = ScxCsr::new_unchecked(
            (3, 5),
            vec![0i64, 2, 3, 5],
            vec![0i32, 2, 1, 0, 4],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
        );
        host.stage(&csr_a).unwrap();
        host.upload_to(dev.stream(), &mut slot, 3, 5, 5).unwrap();
        let p1 = slot.cached_sp_descr(&dev, dev.stream()).unwrap().raw();
        assert!(slot.has_cached_descr());

        // Re-upload a DIFFERENT shape. Cache must invalidate.
        let csr_b = ScxCsr::new_unchecked(
            (4, 5),
            vec![0i64, 1, 2, 3, 4],
            vec![0i32, 1, 2, 3],
            vec![10.0f32, 11.0, 12.0, 13.0],
        );
        host.stage(&csr_b).unwrap();
        host.upload_to(dev.stream(), &mut slot, 4, 4, 5).unwrap();
        // Cache should have been invalidated by the shape change.
        assert!(
            !slot.has_cached_descr(),
            "shape change must invalidate descr cache"
        );
        let p2 = slot.cached_sp_descr(&dev, dev.stream()).unwrap().raw();
        // Different descriptor object after invalidation. (Address may
        // happen to alias if cuSPARSE recycles, but at minimum it was
        // rebuilt — the has_cached_descr assertion above is the load-
        // bearing check; this is a soft sanity check.)
        let _ = (p1, p2);
    }
}
