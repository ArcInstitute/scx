//! cuSPARSE CSR interop for GPU-resident sparse matrices.
//!
//! Provides [`CusparseHandle`] (library handle) and methods on [`GpuCsr`] to
//! create cuSPARSE sparse matrix descriptors (`cusparseSpMatDescr_t`) and
//! expose raw device pointers for cupy `__cuda_array_interface__` interop.

use std::mem::MaybeUninit;

use cudarc::cusparse::sys::{
    self as csp, cudaDataType, cusparseIndexBase_t, cusparseIndexType_t, cusparseSpMatDescr_t,
};
use cudarc::driver::safe::{CudaStream, DevicePtr};

use crate::error::GpuError;
use crate::shard_decode::GpuCsr;

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
}

impl CusparseSpMatDescr {
    /// Access the raw `cusparseSpMatDescr_t` for FFI calls.
    pub fn raw(&self) -> cusparseSpMatDescr_t {
        self.raw
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
    /// Create a cuSPARSE CSR sparse matrix descriptor (zero-copy).
    ///
    /// The descriptor references the existing GPU memory in this `GpuCsr`.
    /// This `GpuCsr` **must outlive** the returned descriptor — the descriptor
    /// holds raw pointers into `self`'s device buffers.
    ///
    /// Type mapping:
    /// - indptr: `CUSPARSE_INDEX_64I` (i64)
    /// - indices: `CUSPARSE_INDEX_32I` (i32)
    /// - data: `CUDA_R_32F` (f32)
    /// - index base: zero-based
    pub fn to_cusparse_csr(&self, stream: &CudaStream) -> Result<CusparseSpMatDescr, GpuError> {
        let (n_rows, n_cols) = self.shape;
        let nnz = self.indices.len();

        // Get raw device pointers. The SyncOnDrop guards ensure proper
        // synchronization — they are dropped at the end of this scope,
        // after cusparseCreateCsr has captured the pointers.
        let (indptr_ptr, _guard_indptr) = self.indptr.device_ptr(stream);
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
                cusparseIndexType_t::CUSPARSE_INDEX_64I,
                cusparseIndexType_t::CUSPARSE_INDEX_32I,
                cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                cudaDataType::CUDA_R_32F,
            )
            .result()
            .map_err(|e| GpuError::CuSparseError(format!("cusparseCreateCsr: {e:?}")))?;

            Ok(CusparseSpMatDescr {
                raw: desc.assume_init(),
            })
        }
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

    macro_rules! require_gpu {
        () => {
            match GpuDevice::new(0) {
                Ok(dev) => dev,
                Err(_) => {
                    eprintln!("CUDA not available — skipping GPU test");
                    return;
                }
            }
        };
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
        let desc = gpu_csr.to_cusparse_csr(dev.stream()).unwrap();

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
}
