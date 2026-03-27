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
//! let a_desc = gpu_csr.to_cusparse_csr(dev.stream())?;
//! // B is column-major dense (k × n), C is column-major dense (m × n)
//! spmm_csr(&handle, dev.stream(), &a_desc, &b_device, &mut c_device,
//!          m, k, n, 1.0, 0.0)?;
//! ```

use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::cusparse::sys::{
    self as csp, cudaDataType, cusparseIndexBase_t, cusparseIndexType_t, cusparseSpMatDescr_t,
};
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::device::GpuDevice;
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

/// Internal SpMM implementation shared by `spmm_csr` and `spmm_csr_transpose`.
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
    // Bind cuSPARSE handle to our CUDA stream
    handle.set_stream(stream)?;

    // Get raw device pointers for B (read) and C (read-write)
    let (b_ptr, _guard_b) = b.device_ptr(stream);
    let (c_ptr, _guard_c) = c.device_ptr_mut(stream);

    // Determine dense matrix dimensions based on the operation.
    // For NON_TRANSPOSE: A is (m×k), B is (k×n), C is (m×n)
    // For TRANSPOSE:     A^T is (k×m), so the original A is (m×k),
    //                    B is (m×n), C is (k×n)
    let (b_rows, b_cols, c_rows, c_cols) = match op_a {
        csp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE => (k, n, m, n),
        _ => (m, n, k, n), // TRANSPOSE or CONJUGATE_TRANSPOSE
    };

    // Create dense matrix descriptors (column-major, leading dim = rows)
    let dn_b = DnMatDescr::new(b_ptr, b_rows as i64, b_cols as i64, b_rows as i64)?;
    let dn_c = DnMatDescr::new(c_ptr, c_rows as i64, c_cols as i64, c_rows as i64)?;

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

    // Allocate workspace (may be 0 bytes for simple cases)
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

    // Execute SpMM
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
        let a_desc = gpu_csr.to_cusparse_csr(dev.stream()).unwrap();

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
        let a_desc = gpu_csr.to_cusparse_csr(dev.stream()).unwrap();

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
        let a_desc = gpu_csr.to_cusparse_csr(dev.stream()).unwrap();

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
