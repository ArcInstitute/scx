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

use cudarc::cublas::sys as cbs;
use cudarc::cusolver::sys as csol;
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtrMut};

use crate::cublas::{gpu_sgemm, gpu_strsm, CublasHandle};
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

// SAFETY: the wrapper owns its raw cuSOLVER handle exclusively, so it is safe to
// MOVE between threads (Send) — only one thread can access it at a time. It is
// deliberately NOT `Sync`: a cuSOLVER handle is stream-bound and NVIDIA requires
// one handle per thread, so sharing `&CusolverHandle` for concurrent calls would
// race on the handle's internal stream/state. Need a handle on another thread?
// Create one there with `CusolverHandle::new()`.
unsafe impl Send for CusolverHandle {}

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

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition (cusolverDnSsyevd)
// ---------------------------------------------------------------------------

/// Dense symmetric eigendecomposition on GPU (single-precision).
///
/// Solves `A · V = V · diag(Λ)` for a real symmetric `A`, in place on `A`:
/// on return, `A` holds the eigenvectors as a column-major `n × n` matrix and
/// the returned `CudaSlice<f32>` holds the eigenvalues.
///
/// Eigenvalues come back in **ascending** order (standard cuSOLVER / LAPACK
/// `ssyevd` behaviour). Callers that want the top-k PCs should take the last
/// `k` eigenpairs and reverse them.
///
/// The input `A` is expected to be symmetric col-major (we only inspect the
/// upper triangle per `CUBLAS_FILL_MODE_UPPER`; the lower triangle is ignored).
///
/// Uses the divide-and-conquer algorithm under the hood.
pub fn gpu_eigh_sym(
    handle: &CusolverHandle,
    stream: &Arc<CudaStream>,
    dev: &GpuDevice,
    a: &mut CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    if n == 0 {
        return Err(GpuError::CuSolverError("eigh requires n > 0".into()));
    }
    if a.len() < n * n {
        return Err(GpuError::CuSolverError(format!(
            "eigh input too small: got {} elements, need {}",
            a.len(),
            n * n
        )));
    }

    let n_i32 = n as i32;
    let lda = n_i32;
    let jobz = csol::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
    let uplo = csol::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;

    handle.set_stream(stream)?;

    let mut eigvals = dev.alloc_zeros::<f32>(n)?;

    // Query workspace size.
    let mut lwork: i32 = 0;
    {
        let (a_ptr, _ga) = a.device_ptr_mut(stream);
        let (w_ptr, _gw) = eigvals.device_ptr_mut(stream);
        unsafe {
            csol::cusolverDnSsyevd_bufferSize(
                handle.raw(),
                jobz,
                uplo,
                n_i32,
                a_ptr as *const f32,
                lda,
                w_ptr as *const f32,
                &mut lwork as *mut i32,
            )
            .result()
            .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSsyevd_bufferSize: {e:?}")))?;
        }
    }

    let mut workspace = dev.alloc_zeros::<f32>(lwork.max(1) as usize)?;
    let mut dev_info = dev.alloc_zeros::<i32>(1)?;

    {
        let (a_ptr, _ga) = a.device_ptr_mut(stream);
        let (w_ptr, _gw) = eigvals.device_ptr_mut(stream);
        let (ws_ptr, _gws) = workspace.device_ptr_mut(stream);
        let (info_ptr, _gi) = dev_info.device_ptr_mut(stream);
        unsafe {
            csol::cusolverDnSsyevd(
                handle.raw(),
                jobz,
                uplo,
                n_i32,
                a_ptr as *mut f32,
                lda,
                w_ptr as *mut f32,
                ws_ptr as *mut f32,
                lwork,
                info_ptr as *mut i32,
            )
            .result()
            .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSsyevd: {e:?}")))?;
        }
    }

    let info_host = dev.dtoh_copy(&dev_info)?;
    if info_host[0] != 0 {
        return Err(GpuError::CuSolverError(format!(
            "cusolverDnSsyevd failed: devInfo = {}",
            info_host[0]
        )));
    }

    Ok(eigvals)
}

// ---------------------------------------------------------------------------
// QR method selector (Phase 4)
// ---------------------------------------------------------------------------

/// QR algorithm choice for the randomized-PCA power iterations.
///
/// - [`QrMethod::Householder`] (default) — uses [`gpu_qr_q`] (`cusolverDnSgeqrf` +
///   `cusolverDnSorgqr`). Slower but numerically the most robust; the fallback
///   when an input is nearly rank-deficient.
/// - [`QrMethod::Cholesky`] — uses [`gpu_cholesky_qr2`] (CholeskyQR2). ~3× faster
///   for the power-iteration QR and produces `Q` orthonormal to f32 precision
///   on well-conditioned inputs. Fails with [`GpuError::CuSolverError`] if the
///   Gram matrix is not positive-definite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QrMethod {
    /// Householder QR via `gpu_qr_q` (default; always-stable).
    #[default]
    Householder,
    /// CholeskyQR2 via `gpu_cholesky_qr2` (opt-in, faster, requires SPD Gram).
    Cholesky,
}

// ---------------------------------------------------------------------------
// CholeskyQR2 (Phase 4)
// ---------------------------------------------------------------------------

/// CholeskyQR2 on GPU — two iterations of Cholesky-based QR.
///
/// Computes `A = Q · R` where `Q` is orthonormal (to f32 precision) and
/// `R` is upper-triangular, overwriting the input buffer `a` with `Q`. The
/// caller receives the same buffer back via an ownership-transfer pattern
/// mirroring [`gpu_qr_q`].
///
/// ## Algorithm
///
/// For `iter in 0..2`:
///   1. `G = Aᵀ A`                   (cuBLAS `sgemm`, trans_a=T, trans_b=N)
///   2. `G = Rᵀ R` (upper triangle)  (`cusolverDnSpotrf`, UPLO=UPPER)
///   3. `A = A · R⁻¹`                (cuBLAS `strsm`, side=RIGHT, uplo=UPPER)
///
/// Two iterations reliably refine `Q` to near-machine f32 orthonormality on
/// well-conditioned inputs (Tomás et al., 2013; used as the default in
/// rapids-singlecell's randomized-PCA path).
///
/// ## Failure
///
/// If `Aᵀ A` is not positive-definite (near-rank-deficient input, condition
/// number > ~1e8 in f32), `cusolverDnSpotrf` reports `devInfo > 0` and this
/// function returns
/// `GpuError::CuSolverError("CholeskyQR2 failed: non-SPD; retry with qr_method='householder'")`.
/// No silent fallback — callers must explicitly switch to Householder.
///
/// ## Requirements
///
/// `m >= k > 0`. The returned `Q` has shape `(m × k)` col-major.
pub fn gpu_cholesky_qr2(
    cublas: &CublasHandle,
    cusolver: &CusolverHandle,
    dev: &GpuDevice,
    a: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    if m == 0 || k == 0 {
        return Err(GpuError::CuSolverError(
            "CholeskyQR2 requires m > 0 and k > 0".into(),
        ));
    }
    if m < k {
        return Err(GpuError::CuSolverError(format!(
            "CholeskyQR2 requires m >= k, got m={m}, k={k}"
        )));
    }

    let stream = dev.stream();

    // Allocate the Gram matrix (k × k) on GPU once, reused across both iterations.
    let mut d_g = dev.alloc_zeros::<f32>(k * k)?;

    // Query potrf workspace size (shape-dependent; k × k identical across both iters).
    let k_i32 = k as i32;
    let uplo = csol::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
    let mut potrf_lwork: i32 = 0;
    {
        let (g_ptr, _gg) = d_g.device_ptr_mut(stream);
        unsafe {
            csol::cusolverDnSpotrf_bufferSize(
                cusolver.raw(),
                uplo,
                k_i32,
                g_ptr as *mut f32,
                k_i32,
                &mut potrf_lwork as *mut i32,
            )
            .result()
            .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSpotrf_bufferSize: {e:?}")))?;
        }
    }
    let mut workspace = dev.alloc_zeros::<f32>(potrf_lwork.max(1) as usize)?;

    for _iter in 0..2 {
        // Step 1: G = Aᵀ · A   (k × k col-major, via sgemm with trans_a=T).
        gpu_sgemm(
            cublas,
            stream,
            a,
            a,
            &mut d_g,
            k,
            k,
            m,
            1.0,
            0.0,
            cbs::cublasOperation_t::CUBLAS_OP_T,
            cbs::cublasOperation_t::CUBLAS_OP_N,
        )?;

        // Step 2: Cholesky G = Rᵀ R (R stored in upper triangle of G).
        cusolver.set_stream(stream)?;
        let mut dev_info = dev.alloc_zeros::<i32>(1)?;
        {
            let (g_ptr, _gg) = d_g.device_ptr_mut(stream);
            let (ws_ptr, _gws) = workspace.device_ptr_mut(stream);
            let (info_ptr, _gi) = dev_info.device_ptr_mut(stream);
            unsafe {
                csol::cusolverDnSpotrf(
                    cusolver.raw(),
                    uplo,
                    k_i32,
                    g_ptr as *mut f32,
                    k_i32,
                    ws_ptr as *mut f32,
                    potrf_lwork,
                    info_ptr as *mut i32,
                )
                .result()
                .map_err(|e| GpuError::CuSolverError(format!("cusolverDnSpotrf: {e:?}")))?;
            }
        }
        let info_host = dev.dtoh_copy(&dev_info)?;
        if info_host[0] != 0 {
            return Err(GpuError::CuSolverError(
                "CholeskyQR2 failed: non-SPD; retry with qr_method='householder'".into(),
            ));
        }

        // Step 3: A = A · R⁻¹  via strsm(side=RIGHT, uplo=UPPER, trans=N, diag=NON_UNIT).
        // Solves X · R = A with X overwriting A; R lives in the upper triangle of d_g.
        gpu_strsm(
            cublas,
            stream,
            &d_g,
            a,
            m,
            k,
            cbs::cublasSideMode_t::CUBLAS_SIDE_RIGHT,
            cbs::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            cbs::cublasOperation_t::CUBLAS_OP_N,
            cbs::cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
            1.0,
        )?;
    }

    // Ownership-transfer — return the input buffer (now Q) to the caller.
    let mut q = dev.alloc_zeros::<f32>(0)?;
    std::mem::swap(a, &mut q);
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_cusolver_handle() {
        let _dev = require_gpu!();
        let handle = CusolverHandle::new().unwrap();
        assert!(!handle.raw().is_null());
        // Drop destroys the handle
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
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
    #[ignore = "requires a CUDA GPU"]
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

    // ---- CholeskyQR2 (Phase 4) ----

    /// Compute ||Q^T Q − I||_F on a col-major `(m × n)` Q downloaded to host.
    fn qtq_minus_i_frobenius(q_host: &[f32], m: usize, n: usize) -> f32 {
        let mut sumsq = 0.0f32;
        for c1 in 0..n {
            for c2 in 0..n {
                let mut dot = 0.0f32;
                for r in 0..m {
                    dot += q_host[c1 * m + r] * q_host[c2 * m + r];
                }
                let target = if c1 == c2 { 1.0 } else { 0.0 };
                let d = dot - target;
                sumsq += d * d;
            }
        }
        sumsq.sqrt()
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_cholesky_qr2_orthonormal() {
        // Phase 4.3 — CQR2 on a well-conditioned 10_000 × 60 Gaussian matrix.
        // Expect ||Q^T Q − I||_F < 1e-4. Random Gaussian matrices have
        // condition number O(sqrt(m/n)) which is far from the SPD failure line.
        use crate::curand::random_gaussian_gpu;

        let dev = require_gpu!();
        let cublas = CublasHandle::new().unwrap();
        let cusolver = CusolverHandle::new().unwrap();

        let m = 10_000;
        let n = 60;
        let mut d_a = random_gaussian_gpu(&dev, dev.stream(), m, n, 42).unwrap();

        let q = gpu_cholesky_qr2(&cublas, &cusolver, &dev, &mut d_a, m, n).unwrap();
        dev.synchronize().unwrap();

        let q_host = dev.dtoh_copy(&q).unwrap();
        assert_eq!(q_host.len(), m * n);
        let err = qtq_minus_i_frobenius(&q_host, m, n);
        assert!(
            err < 1e-4,
            "CQR2 orthonormality: ||Q^T Q - I||_F = {err} (want < 1e-4)"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_cholesky_qr2_ill_conditioned() {
        // Phase 4.4 — deliberately non-SPD input: stack near-duplicate columns
        // (col 0 and col 1 differ by 1e-8-scale noise). A^T A is effectively
        // rank-deficient in f32, so cusolverDnSpotrf reports devInfo > 0 and
        // `gpu_cholesky_qr2` must surface a CuSolverError — no silent fallback.
        let dev = require_gpu!();
        let cublas = CublasHandle::new().unwrap();
        let cusolver = CusolverHandle::new().unwrap();

        let m = 1000;
        let n = 10;
        let mut a_host = vec![0.0f32; m * n];

        // Column 0: deterministic gradient.
        for r in 0..m {
            a_host[r] = (r as f32) * 0.001 + 1.0;
        }
        // Column 1: col 0 + tiny noise (far below f32 precision for col 0's norm).
        for r in 0..m {
            a_host[m + r] = a_host[r] + 1e-8 * ((r % 7) as f32 - 3.0);
        }
        // Columns 2..n: independent, non-pathological — so n_cols > 2 still
        // gets a reasonable SPD matrix if only col 0 / col 1 weren't twins.
        for j in 2..n {
            for r in 0..m {
                a_host[j * m + r] = ((r + j * 17) as f32).sin();
            }
        }

        let mut d_a = dev.htod_copy(&a_host).unwrap();
        let result = gpu_cholesky_qr2(&cublas, &cusolver, &dev, &mut d_a, m, n);
        match result {
            Err(GpuError::CuSolverError(msg)) => {
                assert!(
                    msg.contains("non-SPD"),
                    "expected 'non-SPD' in error message, got: {msg}"
                );
            }
            Err(other) => panic!("expected CuSolverError, got: {other:?}"),
            Ok(_) => panic!("expected CQR2 to fail on near-rank-deficient input"),
        }
    }
}
