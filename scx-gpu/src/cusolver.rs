//! cuSOLVER wrapper for GPU-accelerated dense linear algebra.
//!
//! Provides [`CusolverHandle`] (RAII wrapper) and [`gpu_qr_q`] for economy QR
//! decomposition on GPU. Used in the GPU PCA pipeline for orthogonalizing the
//! streaming SpMM output.
//!
//! ## QR Usage
//!
//! ```ignore
//! let handle = CusolverHandle::new()?;
//! let q = gpu_qr_q(&handle, dev.stream(), &mut a_device, m, n)?;
//! // q is (m × n) col-major, orthonormal columns
//! ```

use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::cusolver::sys as csol;
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtrMut};

use crate::device::GpuDevice;
use crate::error::GpuError;

/// RAII wrapper around a cuSOLVER dense handle (`cusolverDnHandle_t`).
///
/// Created once per device, reusable across multiple QR/SVD calls.
pub struct CusolverHandle {
    raw: csol::cusolverDnHandle_t,
}

impl CusolverHandle {
    /// Create a new cuSOLVER dense handle on the current CUDA context.
    pub fn new() -> Result<Self, GpuError> {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            csol::cusolverDnCreate(handle.as_mut_ptr())
                .result()
                .map_err(|e| GpuError::CuSolverError(format!("cusolverDnCreate: {e:?}")))?;
            Ok(Self {
                raw: handle.assume_init(),
            })
        }
    }

    /// Access the raw `cusolverDnHandle_t` for FFI calls.
    pub fn raw(&self) -> csol::cusolverDnHandle_t {
        self.raw
    }

    /// Bind this handle to the given CUDA stream.
    fn set_stream(&self, stream: &CudaStream) -> Result<(), GpuError> {
        unsafe {
            csol::cusolverDnSetStream(self.raw, stream.cu_stream() as _)
                .result()
                .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSetStream: {e:?}")))
        }
    }
}

impl Drop for CusolverHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = csol::cusolverDnDestroy(self.raw);
        }
    }
}

// SAFETY: cuSOLVER handles are thread-safe per NVIDIA documentation.
unsafe impl Send for CusolverHandle {}
unsafe impl Sync for CusolverHandle {}

/// Economy QR decomposition on GPU: A = Q·R, returning Q.
///
/// - `a`: col-major `[m × n]` on GPU, **overwritten** during factorization
/// - Returns Q: `CudaSlice<f32>` col-major `[m × n]` with orthonormal columns
///
/// Uses `cusolverDnSgeqrf` (Householder QR) + `cusolverDnSorgqr` (construct Q).
///
/// For PCA: m = n_obs (potentially millions), n = k (60). The large m makes
/// GPU worthwhile despite the relatively small n.
pub fn gpu_qr_q(
    handle: &CusolverHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    a: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    if m == 0 || n == 0 {
        return Err(GpuError::CuSolverError(
            "QR requires m > 0 and n > 0".into(),
        ));
    }
    if m < n {
        return Err(GpuError::CuSolverError(format!(
            "QR requires m >= n, got m={m}, n={n}"
        )));
    }

    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let lda = m_i32; // col-major: leading dim = m

    handle.set_stream(stream)?;

    // --- Step 1: cusolverDnSgeqrf (Householder QR factorization) ---

    // Query workspace size
    let mut geqrf_lwork: i32 = 0;
    {
        let (a_ptr, _guard_a) = a.device_ptr_mut(stream);
        unsafe {
            csol::cusolverDnSgeqrf_bufferSize(
                handle.raw(),
                m_i32,
                n_i32,
                a_ptr as *mut f32,
                lda,
                &mut geqrf_lwork as *mut i32,
            )
            .result()
            .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSgeqrf_bufferSize: {e:?}")))?;
        }
    }

    // Allocate workspace, tau vector, and devInfo
    let mut workspace = dev.alloc_zeros::<f32>(geqrf_lwork as usize)?;
    let mut tau = dev.alloc_zeros::<f32>(n)?;
    let mut dev_info = dev.alloc_zeros::<i32>(1)?;

    // Execute QR factorization (overwrites A with R in upper triangle + Householder reflectors)
    {
        let (a_ptr, _ga) = a.device_ptr_mut(stream);
        let (tau_ptr, _gt) = tau.device_ptr_mut(stream);
        let (ws_ptr, _gw) = workspace.device_ptr_mut(stream);
        let (info_ptr, _gi) = dev_info.device_ptr_mut(stream);
        unsafe {
            csol::cusolverDnSgeqrf(
                handle.raw(),
                m_i32,
                n_i32,
                a_ptr as *mut f32,
                lda,
                tau_ptr as *mut f32,
                ws_ptr as *mut f32,
                geqrf_lwork,
                info_ptr as *mut i32,
            )
            .result()
            .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSgeqrf: {e:?}")))?;
        }
    }

    // Check devInfo
    let info_host = dev.dtoh_copy(&dev_info)?;
    if info_host[0] != 0 {
        return Err(GpuError::CuSolverError(format!(
            "cusolverDnSgeqrf failed: devInfo = {}",
            info_host[0]
        )));
    }

    // --- Step 2: cusolverDnSorgqr (construct Q from Householder reflectors) ---

    // Query workspace size for orgqr
    let mut orgqr_lwork: i32 = 0;
    {
        let (a_ptr, _ga) = a.device_ptr_mut(stream);
        let (tau_ptr, _gt) = tau.device_ptr_mut(stream);
        unsafe {
            csol::cusolverDnSorgqr_bufferSize(
                handle.raw(),
                m_i32,
                n_i32,
                n_i32, // k = n (economy QR: all n reflectors)
                a_ptr as *const f32,
                lda,
                tau_ptr as *const f32,
                &mut orgqr_lwork as *mut i32,
            )
            .result()
            .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSorgqr_bufferSize: {e:?}")))?;
        }
    }

    // Allocate workspace for orgqr
    let mut workspace2 = dev.alloc_zeros::<f32>(orgqr_lwork as usize)?;
    let mut dev_info2 = dev.alloc_zeros::<i32>(1)?;

    // Construct Q in-place (overwrites A with the first n columns of Q)
    {
        let (a_ptr, _ga) = a.device_ptr_mut(stream);
        let (tau_ptr, _gt) = tau.device_ptr_mut(stream);
        let (ws2_ptr, _gw) = workspace2.device_ptr_mut(stream);
        let (info2_ptr, _gi) = dev_info2.device_ptr_mut(stream);
        unsafe {
            csol::cusolverDnSorgqr(
                handle.raw(),
                m_i32,
                n_i32,
                n_i32, // k = n
                a_ptr as *mut f32,
                lda,
                tau_ptr as *const f32,
                ws2_ptr as *mut f32,
                orgqr_lwork,
                info2_ptr as *mut i32,
            )
            .result()
            .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSorgqr: {e:?}")))?;
        }
    }

    // Check devInfo
    let info2_host = dev.dtoh_copy(&dev_info2)?;
    if info2_host[0] != 0 {
        return Err(GpuError::CuSolverError(format!(
            "cusolverDnSorgqr failed: devInfo = {}",
            info2_host[0]
        )));
    }

    // A now contains Q (m × n, col-major). Return by taking ownership.
    // Use std::mem::swap to transfer GPU allocation without any host round-trip.
    // The caller's `a` is left with a zero-length dummy allocation.
    let mut q = dev.alloc_zeros::<f32>(0)?;
    std::mem::swap(a, &mut q);

    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;

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

    #[test]
    fn test_cusolver_handle() {
        let _dev = require_gpu!();
        let handle = CusolverHandle::new().unwrap();
        assert!(!handle.raw().is_null());
        // Drop destroys the handle
    }

    #[test]
    fn test_gpu_qr_orthonormal() {
        let dev = require_gpu!();
        let handle = CusolverHandle::new().unwrap();

        let m = 6;
        let n = 3;

        // A (col-major 6×3):
        // col 0: [1, 0, 0, 1, 0, 0]
        // col 1: [0, 1, 0, 0, 1, 0]
        // col 2: [0, 0, 1, 0, 0, 1]
        let a_host: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, // col 1
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // col 2
        ];
        let mut d_a = dev.htod_copy(&a_host).unwrap();

        let q = gpu_qr_q(&handle, dev.stream(), &dev, &mut d_a, m, n).unwrap();
        dev.synchronize().unwrap();

        let q_host = dev.dtoh_copy(&q).unwrap();
        assert_eq!(q_host.len(), m * n);

        // Verify Q^T @ Q ≈ I (orthonormal columns)
        // Q is col-major: Q[i,j] = q_host[j * m + i]
        for c1 in 0..n {
            for c2 in 0..n {
                let mut dot: f32 = 0.0;
                for r in 0..m {
                    dot += q_host[c1 * m + r] * q_host[c2 * m + r];
                }
                let expected = if c1 == c2 { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "Q^TQ[{c1},{c2}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_gpu_qr_tall_skinny() {
        let dev = require_gpu!();
        let handle = CusolverHandle::new().unwrap();

        // Tall-skinny matrix: 100 × 5
        let m = 100;
        let n = 5;
        let mut a_host: Vec<f32> = vec![0.0; m * n];
        // Fill with a simple pattern: A[i, j] = (i * n + j + 1) as f32
        for j in 0..n {
            for i in 0..m {
                a_host[j * m + i] = (i * n + j + 1) as f32;
            }
        }

        let mut d_a = dev.htod_copy(&a_host).unwrap();
        let q = gpu_qr_q(&handle, dev.stream(), &dev, &mut d_a, m, n).unwrap();
        dev.synchronize().unwrap();

        let q_host = dev.dtoh_copy(&q).unwrap();

        // Verify Q^T @ Q ≈ I
        for c1 in 0..n {
            for c2 in 0..n {
                let mut dot: f32 = 0.0;
                for r in 0..m {
                    dot += q_host[c1 * m + r] * q_host[c2 * m + r];
                }
                let expected = if c1 == c2 { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-3,
                    "Q^TQ[{c1},{c2}] = {dot}, expected {expected}"
                );
            }
        }
    }
}
