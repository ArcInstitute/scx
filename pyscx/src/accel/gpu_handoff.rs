//! GPU shard → cuPy CSR handoff seam (ACC-RUST-OPT-V4 Phase 0.4).
//!
//! Phase 0.4 proves that a decoded SCX shard's **device** buffers can back a
//! `cupyx.scipy.sparse.csr_matrix` that rapids-singlecell consumes **without a
//! copy** — the minimal-copy mechanism the §2.5 `to_gpu_anndata` (Phase 1.2)
//! design is built on. This module is that primitive: it decodes one CSR shard
//! GPU-side via `scx_gpu::decode_shard_gpu` and exposes the resulting
//! `data` / `indices` / `indptr` device arrays through the
//! `__cuda_array_interface__` protocol so cuPy can adopt them in place.
//!
//! Lifetime: [`GpuShardCsr`] owns the decoded [`scx_gpu::GpuCsr`], keeping the
//! device allocations alive. Each array view holds a reference back to its
//! parent `GpuShardCsr`, so a cuPy array created via `cupy.asarray(view)` (which
//! stores the view as its `.base`) transitively keeps the device memory alive.
//!
//! The classes are `unsendable`: the underlying `GpuDevice` is not `Send`
//! (its module cache keys on `*const str`), so a `GpuShardCsr` must be used on
//! the thread that created it. That is fine for the GIL-bound PoC.

#[cfg(feature = "gpu")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "gpu")]
use pyo3::prelude::*;
#[cfg(feature = "gpu")]
use pyo3::types::PyDict;

/// A single GPU-resident array (`data`, `indices`, or `indptr`) exposed via the
/// `__cuda_array_interface__` (CAI) protocol for zero-copy adoption by cuPy.
#[cfg(feature = "gpu")]
#[pyclass(unsendable, name = "CudaArrayView", module = "pyscx.accel")]
pub struct CudaArrayView {
    /// Keeps the owning shard (and thus the device memory) alive.
    _parent: Py<GpuShardCsr>,
    ptr: u64,
    len: usize,
    /// numpy typestr, e.g. "<f4" / "<i4" / "<i8".
    typestr: &'static str,
}

#[cfg(feature = "gpu")]
#[pymethods]
impl CudaArrayView {
    /// The CUDA Array Interface (version 3) describing this 1-D device array.
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.len,))?;
        d.set_item("typestr", self.typestr)?;
        // (device pointer, read_only=false). cuPy adopts this without copying.
        d.set_item("data", (self.ptr, false))?;
        d.set_item("version", 3)?;
        d.set_item("strides", py.None())?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.len
    }

    fn __repr__(&self) -> String {
        format!(
            "CudaArrayView(len={}, typestr='{}', ptr=0x{:x})",
            self.len, self.typestr, self.ptr
        )
    }
}

/// A GPU-resident CSR shard whose `data` / `indices` / `indptr` device buffers
/// can be adopted by `cupyx.scipy.sparse.csr_matrix` with no host round-trip.
///
/// Construct via [`gpu_decode_shard`]. Intended for the Phase 0.4 PoC and as the
/// Phase 1.2 `to_gpu_anndata` building block — not a stable public API yet.
#[cfg(feature = "gpu")]
#[pyclass(unsendable, name = "GpuShardCsr", module = "pyscx.accel")]
pub struct GpuShardCsr {
    // Field order matters for drop: `csr` (device buffers) must drop before
    // `_dev` (context/stream). Rust drops fields in declaration order.
    // Held only to own the device allocations the CAI pointers reference.
    #[allow(dead_code)]
    csr: scx_accel::GpuCsr,
    _dev: scx_accel::GpuDevice,
    data_ptr: u64,
    indices_ptr: u64,
    indptr_ptr: u64,
    n_rows: usize,
    n_cols: usize,
    nnz: usize,
}

#[cfg(feature = "gpu")]
#[pymethods]
impl GpuShardCsr {
    /// Matrix shape `(n_rows, n_cols)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Number of stored non-zeros.
    #[getter]
    fn nnz(&self) -> usize {
        self.nnz
    }

    /// `data` array (f32, length `nnz`) as a CAI view.
    fn data(slf: Bound<'_, Self>) -> PyResult<CudaArrayView> {
        let (ptr, len) = {
            let b = slf.borrow();
            (b.data_ptr, b.nnz)
        };
        Ok(CudaArrayView {
            _parent: slf.unbind(),
            ptr,
            len,
            typestr: "<f4",
        })
    }

    /// `indices` array (i32, length `nnz`) as a CAI view.
    fn indices(slf: Bound<'_, Self>) -> PyResult<CudaArrayView> {
        let (ptr, len) = {
            let b = slf.borrow();
            (b.indices_ptr, b.nnz)
        };
        Ok(CudaArrayView {
            _parent: slf.unbind(),
            ptr,
            len,
            typestr: "<i4",
        })
    }

    /// `indptr` array (i64, length `n_rows + 1`) as a CAI view.
    fn indptr(slf: Bound<'_, Self>) -> PyResult<CudaArrayView> {
        let (ptr, len) = {
            let b = slf.borrow();
            (b.indptr_ptr, b.n_rows + 1)
        };
        Ok(CudaArrayView {
            _parent: slf.unbind(),
            ptr,
            len,
            typestr: "<i8",
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuShardCsr(shape=({}, {}), nnz={})",
            self.n_rows, self.n_cols, self.nnz
        )
    }
}

/// Decode one CSR shard of an SCX file directly onto the GPU and return a
/// [`GpuShardCsr`] whose device buffers can back a `cupyx` CSR with no copy.
///
/// `shard_idx` is the 0-based index in catalog order within `modality_id`
/// (default modality 0). `device` accepts `"gpu"` / `"gpu:N"` / `"auto"`; `"cpu"`
/// is rejected (there is nothing to hand off). Scx1 shards decode GPU-side
/// (Rice/FOR-BP); Zstd/Pcodec/LZ4/None shards decode on the host and incur one
/// HtoD upload — set `SCX_GPU_PROFILE=1` and read `gpu_profile_snapshot()` to
/// see the per-codec split.
#[cfg(feature = "gpu")]
#[pyfunction]
#[pyo3(signature = (path, shard_idx=0, device="gpu", modality_id=0))]
pub fn gpu_decode_shard(
    path: &str,
    shard_idx: usize,
    device: &str,
    modality_id: u8,
) -> PyResult<GpuShardCsr> {
    {
        use super::gpu::resolve_device;

        let resolved = resolve_device(device)?;
        let gpu_id = resolved.gpu_id().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "gpu_decode_shard requires a GPU device ('gpu', 'gpu:N', or 'auto' on a GPU host)",
            )
        })?;

        let reader = scx_format::ScxReader::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open '{path}': {e}")))?;
        let shard_bytes = reader
            .read_raw_csr_shard_bytes_for(modality_id, shard_idx)
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "failed to read CSR shard {shard_idx} (modality {modality_id}): {e}"
                ))
            })?;

        let dev = scx_accel::GpuDevice::new(gpu_id)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU device {gpu_id}: {e}")))?;
        let csr = scx_accel::decode_shard_gpu(&dev, shard_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU shard decode failed: {e}")))?;
        // The Scx1 path launches async Rice/FOR-BP/cast kernels on the stream.
        // Synchronize before exposing device pointers so a cuPy consumer cannot
        // race the still-running decode (the handoff crosses to a different
        // consumer, unlike internal callers that chain on the same stream).
        dev.synchronize()
            .map_err(|e| PyRuntimeError::new_err(format!("GPU decode synchronize: {e}")))?;

        let (n_rows, n_cols) = csr.shape;
        let nnz = csr.indices.len();
        let ptrs = csr.device_pointers(dev.stream());

        Ok(GpuShardCsr {
            csr,
            _dev: dev,
            data_ptr: ptrs.data_ptr,
            indices_ptr: ptrs.indices_ptr,
            indptr_ptr: ptrs.indptr_ptr,
            n_rows,
            n_cols,
            nnz,
        })
    }
}
