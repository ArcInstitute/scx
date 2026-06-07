//! cuBLAS wrapper for GPU-accelerated dense linear algebra.
//!
//! Provides [`CublasHandle`] (RAII wrapper) and safe wrappers over
//! `cublas<t>gemm`, `cublas<t>gemv`, `cublas<t>ger`, `cublas<t>trsm`.
//! All dense matrices are column-major (cuBLAS native layout).
//!
//! Consumed by:
//! - `linear_operator::CenteredSparseOperator::accumulate_gram` (via `gpu_sgemm` + `gpu_sger`).
//! - Phase 2 GPU covariance PCA's `d_out += −n · μ μᵀ` correction (`gpu_sger`).
//! - Phase 3's GPU-resident final embedding multiply (`gpu_sgemm`).
//! - Phase 3's `mc = Vᵀ · μ` precomputation (`gpu_sgemv`).
//! - Phase 4 CholeskyQR2 (`gpu_sgemm` + `gpu_strsm`).

use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::cublas::sys as cbs;
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::error::GpuError;

/// RAII wrapper around a cuBLAS library handle (`cublasHandle_t`).
///
/// Created once per device, reusable across multiple GEMM/GEMV/GER/TRSM calls.
/// The handle is bound to the target stream before each op via `set_stream`.
pub struct CublasHandle {
    raw: cbs::cublasHandle_t,
}

impl CublasHandle {
    /// Create a new cuBLAS handle on the current CUDA context.
    pub fn new() -> Result<Self, GpuError> {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            cbs::cublasCreate_v2(handle.as_mut_ptr())
                .result()
                .map_err(|e| GpuError::CuBlasError(format!("cublasCreate: {e:?}")))?;
            Ok(Self {
                raw: handle.assume_init(),
            })
        }
    }

    /// Access the raw `cublasHandle_t` for FFI calls.
    pub fn raw(&self) -> cbs::cublasHandle_t {
        self.raw
    }

    /// Set the cuBLAS math mode (Task 2.5).
    ///
    /// Sticky on the handle until changed: applies to every subsequent GEMM /
    /// GEMV / GER on this handle. [`GpuMathMode::StrictFp32`] maps to
    /// `CUBLAS_DEFAULT_MATH` (full fp32; cuBLAS SGEMM does not use TF32 here),
    /// [`GpuMathMode::AllowTf32`] to `CUBLAS_TF32_TENSOR_OP_MATH`.
    pub fn set_math_mode(&self, mode: crate::math_policy::GpuMathMode) -> Result<(), GpuError> {
        unsafe {
            cbs::cublasSetMathMode(self.raw, mode.to_cublas())
                .result()
                .map_err(|e| GpuError::CuBlasError(format!("cublasSetMathMode: {e:?}")))
        }
    }

    /// Bind this handle to the given CUDA stream.
    ///
    /// All subsequent cuBLAS operations using this handle execute on `stream`.
    fn set_stream(&self, stream: &CudaStream) -> Result<(), GpuError> {
        unsafe {
            cbs::cublasSetStream_v2(self.raw, stream.cu_stream() as _)
                .result()
                .map_err(|e| GpuError::CuBlasError(format!("cublasSetStream: {e:?}")))
        }
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = cbs::cublasDestroy_v2(self.raw);
        }
    }
}

// SAFETY: cuBLAS handles are thread-safe per NVIDIA documentation, provided
// concurrent calls target different streams. The handle internally synchronizes
// its shared state.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

/// Single-precision matrix–matrix multiply: `C = α · op(A) · op(B) + β · C`.
///
/// All matrices are **column-major** with leading dimensions equal to their row count:
/// - `A`: col-major, logical shape `(m, k)` after `trans_a`, backing buffer shape unchanged.
/// - `B`: col-major, logical shape `(k, n)` after `trans_b`.
/// - `C`: col-major, logical shape `(m, n)`.
///
/// `trans_a` / `trans_b` choose whether to interpret A / B as their transposes.
/// When `trans_a == N`, backing shape is `(m, k)` with `lda = m`. When `trans_a == T`,
/// backing shape is `(k, m)` with `lda = k`.
#[allow(clippy::too_many_arguments)]
pub fn gpu_sgemm(
    handle: &CublasHandle,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    trans_a: cbs::cublasOperation_t,
    trans_b: cbs::cublasOperation_t,
) -> Result<(), GpuError> {
    handle.set_stream(stream)?;

    let lda = match trans_a {
        cbs::cublasOperation_t::CUBLAS_OP_N => m,
        _ => k,
    } as i32;
    let ldb = match trans_b {
        cbs::cublasOperation_t::CUBLAS_OP_N => k,
        _ => n,
    } as i32;
    let ldc = m as i32;

    let (a_ptr, _ga) = a.device_ptr(stream);
    let (b_ptr, _gb) = b.device_ptr(stream);
    let (c_ptr, _gc) = c.device_ptr_mut(stream);

    unsafe {
        cbs::cublasSgemm_v2(
            handle.raw(),
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            &alpha as *const f32,
            a_ptr as *const f32,
            lda,
            b_ptr as *const f32,
            ldb,
            &beta as *const f32,
            c_ptr as *mut f32,
            ldc,
        )
        .result()
        .map_err(|e| GpuError::CuBlasError(format!("cublasSgemm: {e:?}")))?;
    }
    Ok(())
}

/// Transpose a column-major `(n_rows × n_cols)` matrix into a row-major
/// `(n_rows × n_cols)` matrix (device → device), via `cublasSgeam`.
///
/// A row-major `(n_rows × n_cols)` buffer is bit-identical to a column-major
/// `(n_cols × n_rows)` buffer, so this computes `C = Aᵀ` where `A` is the
/// col-major source `(n_rows × n_cols, lda = n_rows)` and `C` is col-major
/// `(n_cols × n_rows, ldc = n_cols)`. The result `dst` therefore reads back as
/// the row-major source.
///
/// Used by the device-returning PCA path to turn the col-major embedding `d_u`
/// into a row-major [`crate::DeviceEmbedding`] without a host round-trip.
/// `src` and `dst` must both have length `n_rows * n_cols` and must not alias.
pub fn gpu_transpose_f32(
    handle: &CublasHandle,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), GpuError> {
    // Guard against a mis-sized caller before issuing a raw device-pointer SpMM
    // (cublasSgeam reads/writes exactly `n_rows * n_cols` elements off each
    // pointer; a short buffer would be an out-of-bounds device access).
    let expected_len = n_rows * n_cols;
    if src.len() != expected_len {
        return Err(GpuError::ShapeMismatch {
            expected: format!("src length = n_rows × n_cols = {expected_len}"),
            got: format!("{}", src.len()),
        });
    }
    if dst.len() != expected_len {
        return Err(GpuError::ShapeMismatch {
            expected: format!("dst length = n_rows × n_cols = {expected_len}"),
            got: format!("{}", dst.len()),
        });
    }

    handle.set_stream(stream)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    // C (col-major) is (m × n) = (n_cols × n_rows). op(A) = Aᵀ supplies it.
    let m = n_cols as i32;
    let n = n_rows as i32;
    let lda = n_rows as i32; // A col-major (n_rows × n_cols)
    let ldb = n_cols as i32; // B unused (beta = 0) but must be a valid leading dim
    let ldc = n_cols as i32; // C col-major (n_cols × n_rows)

    let (src_ptr, _gs) = src.device_ptr(stream);
    let (dst_ptr, _gd) = dst.device_ptr_mut(stream);

    unsafe {
        cbs::cublasSgeam(
            handle.raw(),
            cbs::cublasOperation_t::CUBLAS_OP_T,
            cbs::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            &alpha as *const f32,
            src_ptr as *const f32,
            lda,
            &beta as *const f32,
            // B is ignored because beta = 0; reuse src as a valid pointer.
            src_ptr as *const f32,
            ldb,
            dst_ptr as *mut f32,
            ldc,
        )
        .result()
        .map_err(|e| GpuError::CuBlasError(format!("cublasSgeam (transpose): {e:?}")))?;
    }
    Ok(())
}

/// Single-precision matrix–vector multiply: `y = α · op(A) · x + β · y`.
///
/// `A` is col-major with backing shape `(m, n)` and `lda = m`. When `trans == N`,
/// `x` has length `n` and `y` has length `m`. When `trans == T`, the roles swap
/// (`x` has length `m`, `y` has length `n`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_sgemv(
    handle: &CublasHandle,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    trans: cbs::cublasOperation_t,
) -> Result<(), GpuError> {
    handle.set_stream(stream)?;

    let lda = m as i32;
    let (a_ptr, _ga) = a.device_ptr(stream);
    let (x_ptr, _gx) = x.device_ptr(stream);
    let (y_ptr, _gy) = y.device_ptr_mut(stream);

    unsafe {
        cbs::cublasSgemv_v2(
            handle.raw(),
            trans,
            m as i32,
            n as i32,
            &alpha as *const f32,
            a_ptr as *const f32,
            lda,
            x_ptr as *const f32,
            1,
            &beta as *const f32,
            y_ptr as *mut f32,
            1,
        )
        .result()
        .map_err(|e| GpuError::CuBlasError(format!("cublasSgemv: {e:?}")))?;
    }
    Ok(())
}

/// Rank-1 update: `A = α · x · yᵀ + A`.
///
/// `A` is col-major with shape `(m, n)` and `lda = m`. `x` has length `m`,
/// `y` has length `n`. Used in the covariance-PCA Gram correction
/// (`Gram += −n · μ μᵀ`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_sger(
    handle: &CublasHandle,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    a: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    alpha: f32,
) -> Result<(), GpuError> {
    handle.set_stream(stream)?;

    let lda = m as i32;
    let (x_ptr, _gx) = x.device_ptr(stream);
    let (y_ptr, _gy) = y.device_ptr(stream);
    let (a_ptr, _ga) = a.device_ptr_mut(stream);

    unsafe {
        cbs::cublasSger_v2(
            handle.raw(),
            m as i32,
            n as i32,
            &alpha as *const f32,
            x_ptr as *const f32,
            1,
            y_ptr as *const f32,
            1,
            a_ptr as *mut f32,
            lda,
        )
        .result()
        .map_err(|e| GpuError::CuBlasError(format!("cublasSger: {e:?}")))?;
    }
    Ok(())
}

/// Triangular solve: solves `op(A) · X = α · B` (`side = LEFT`) or
/// `X · op(A) = α · B` (`side = RIGHT`), overwriting `B` with `X`.
///
/// `A` is triangular (`uplo == UPPER` or `LOWER`) with an implicit/explicit unit
/// diagonal (`diag`). `B` / `X` is col-major `(m, n)`. Used in CholeskyQR2.
#[allow(clippy::too_many_arguments)]
pub fn gpu_strsm(
    handle: &CublasHandle,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    b: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    side: cbs::cublasSideMode_t,
    uplo: cbs::cublasFillMode_t,
    trans: cbs::cublasOperation_t,
    diag: cbs::cublasDiagType_t,
    alpha: f32,
) -> Result<(), GpuError> {
    handle.set_stream(stream)?;

    // For SIDE_LEFT: A is m×m, lda = m. For SIDE_RIGHT: A is n×n, lda = n.
    let lda = match side {
        cbs::cublasSideMode_t::CUBLAS_SIDE_LEFT => m,
        _ => n,
    } as i32;
    let ldb = m as i32;

    let (a_ptr, _ga) = a.device_ptr(stream);
    let (b_ptr, _gb) = b.device_ptr_mut(stream);

    unsafe {
        cbs::cublasStrsm_v2(
            handle.raw(),
            side,
            uplo,
            trans,
            diag,
            m as i32,
            n as i32,
            &alpha as *const f32,
            a_ptr as *const f32,
            lda,
            b_ptr as *mut f32,
            ldb,
        )
        .result()
        .map_err(|e| GpuError::CuBlasError(format!("cublasStrsm: {e:?}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cublas_handle() {
        let _dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();
        assert!(!handle.raw().is_null());
    }

    /// CPU reference GEMM (column-major): `C = α · op(A) · op(B) + β · C`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_sgemm(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    ) {
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let a_val = if trans_a {
                        a[i * k + p] // A^T: logical (m,k), backing (k,m) col-major → A^T[i,p] = A[p,i] = a[i*k + p]
                    } else {
                        a[p * m + i] // A: backing (m,k) col-major → A[i,p] = a[p*m + i]
                    };
                    let b_val = if trans_b { b[p * n + j] } else { b[j * k + p] };
                    acc += a_val * b_val;
                }
                c[j * m + i] = alpha * acc + beta * c[j * m + i];
            }
        }
    }

    #[test]
    fn test_sgemm_matches_cpu() {
        let dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();

        // A (col-major 4×3), B (col-major 3×2), C (col-major 4×2). No transpose.
        let m = 4;
        let n = 2;
        let k = 3;
        let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let c_init: Vec<f32> = vec![0.0; m * n];

        let d_a = dev.htod_copy(&a_host).unwrap();
        let d_b = dev.htod_copy(&b_host).unwrap();
        let mut d_c = dev.htod_copy(&c_init).unwrap();

        gpu_sgemm(
            &handle,
            dev.stream(),
            &d_a,
            &d_b,
            &mut d_c,
            m,
            n,
            k,
            1.0,
            0.0,
            cbs::cublasOperation_t::CUBLAS_OP_N,
            cbs::cublasOperation_t::CUBLAS_OP_N,
        )
        .unwrap();
        dev.synchronize().unwrap();
        let c_gpu = dev.dtoh_copy(&d_c).unwrap();

        let mut c_cpu = c_init.clone();
        cpu_sgemm(
            &a_host, &b_host, &mut c_cpu, m, n, k, 1.0, 0.0, false, false,
        );

        for i in 0..c_gpu.len() {
            assert!(
                (c_gpu[i] - c_cpu[i]).abs() < 1e-4,
                "GEMM mismatch at {i}: gpu={}, cpu={}",
                c_gpu[i],
                c_cpu[i]
            );
        }
    }

    #[test]
    fn test_sgemm_transpose_a() {
        let dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();

        // Aᵀ: logical (m, k), backing (k, m). Gram-matrix pattern.
        let m = 3; // logical rows of op(A) = n_vars
        let n = 3;
        let k = 5; // logical cols of op(A) = shard_rows
        let a_host: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.13 + 0.2).collect();
        let b_host: Vec<f32> = a_host.clone(); // B same backing as A for Gram-like op.
        let c_init = vec![0.5f32; m * n];

        let d_a = dev.htod_copy(&a_host).unwrap();
        let d_b = dev.htod_copy(&b_host).unwrap();
        let mut d_c = dev.htod_copy(&c_init).unwrap();

        gpu_sgemm(
            &handle,
            dev.stream(),
            &d_a,
            &d_b,
            &mut d_c,
            m,
            n,
            k,
            1.0,
            1.0, // beta = 1 (accumulate)
            cbs::cublasOperation_t::CUBLAS_OP_T,
            cbs::cublasOperation_t::CUBLAS_OP_N,
        )
        .unwrap();
        dev.synchronize().unwrap();
        let c_gpu = dev.dtoh_copy(&d_c).unwrap();

        let mut c_cpu = c_init.clone();
        cpu_sgemm(&a_host, &b_host, &mut c_cpu, m, n, k, 1.0, 1.0, true, false);

        for i in 0..c_gpu.len() {
            assert!(
                (c_gpu[i] - c_cpu[i]).abs() < 1e-4,
                "GEMM(Aᵀ) mismatch at {i}: gpu={}, cpu={}",
                c_gpu[i],
                c_cpu[i]
            );
        }
    }

    #[test]
    fn test_sgemv_matches_cpu() {
        let dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();

        // A: col-major 5×3, x: 3-vec, y: 5-vec. trans_a = N → y = A @ x.
        let m = 5;
        let n = 3;
        let a_host: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.07 + 0.1).collect();
        let x_host: Vec<f32> = vec![1.0, -2.0, 3.5];
        let y_init = vec![0.0f32; m];

        let d_a = dev.htod_copy(&a_host).unwrap();
        let d_x = dev.htod_copy(&x_host).unwrap();
        let mut d_y = dev.htod_copy(&y_init).unwrap();

        gpu_sgemv(
            &handle,
            dev.stream(),
            &d_a,
            &d_x,
            &mut d_y,
            m,
            n,
            1.0,
            0.0,
            cbs::cublasOperation_t::CUBLAS_OP_N,
        )
        .unwrap();
        dev.synchronize().unwrap();
        let y_gpu = dev.dtoh_copy(&d_y).unwrap();

        // CPU ref: y[i] = Σ_j a[j*m + i] * x[j]
        let mut y_cpu = y_init.clone();
        for i in 0..m {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += a_host[j * m + i] * x_host[j];
            }
            y_cpu[i] = acc;
        }

        for i in 0..m {
            assert!(
                (y_gpu[i] - y_cpu[i]).abs() < 1e-4,
                "GEMV mismatch at {i}: gpu={}, cpu={}",
                y_gpu[i],
                y_cpu[i]
            );
        }
    }

    #[test]
    fn test_sgemv_transpose() {
        let dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();

        // Aᵀ @ x: A is (m=5, n=3) col-major, x is m-vec, y is n-vec.
        let m = 5;
        let n = 3;
        let a_host: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.07 + 0.1).collect();
        let x_host: Vec<f32> = vec![1.0, -1.0, 0.5, 2.0, -0.25];
        let y_init = vec![0.0f32; n];

        let d_a = dev.htod_copy(&a_host).unwrap();
        let d_x = dev.htod_copy(&x_host).unwrap();
        let mut d_y = dev.htod_copy(&y_init).unwrap();

        gpu_sgemv(
            &handle,
            dev.stream(),
            &d_a,
            &d_x,
            &mut d_y,
            m,
            n,
            1.0,
            0.0,
            cbs::cublasOperation_t::CUBLAS_OP_T,
        )
        .unwrap();
        dev.synchronize().unwrap();
        let y_gpu = dev.dtoh_copy(&d_y).unwrap();

        // CPU ref: y[j] = Σ_i a[j*m + i] * x[i]
        let mut y_cpu = y_init.clone();
        for j in 0..n {
            let mut acc = 0.0f32;
            for i in 0..m {
                acc += a_host[j * m + i] * x_host[i];
            }
            y_cpu[j] = acc;
        }

        for j in 0..n {
            assert!(
                (y_gpu[j] - y_cpu[j]).abs() < 1e-4,
                "GEMV(Aᵀ) mismatch at {j}: gpu={}, cpu={}",
                y_gpu[j],
                y_cpu[j]
            );
        }
    }

    #[test]
    fn test_sger_matches_cpu() {
        let dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();

        // A: col-major 4×3, x: 4-vec, y: 3-vec. A += α · x · yᵀ.
        let m = 4;
        let n = 3;
        let a_init: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
        let x_host = vec![1.0f32, 2.0, 3.0, -1.0];
        let y_host = vec![0.5f32, -0.5, 1.0];
        let alpha = -2.0f32;

        let d_x = dev.htod_copy(&x_host).unwrap();
        let d_y = dev.htod_copy(&y_host).unwrap();
        let mut d_a = dev.htod_copy(&a_init).unwrap();

        gpu_sger(&handle, dev.stream(), &d_x, &d_y, &mut d_a, m, n, alpha).unwrap();
        dev.synchronize().unwrap();
        let a_gpu = dev.dtoh_copy(&d_a).unwrap();

        let mut a_cpu = a_init.clone();
        for j in 0..n {
            for i in 0..m {
                a_cpu[j * m + i] += alpha * x_host[i] * y_host[j];
            }
        }

        for i in 0..a_gpu.len() {
            assert!(
                (a_gpu[i] - a_cpu[i]).abs() < 1e-4,
                "GER mismatch at {i}: gpu={}, cpu={}",
                a_gpu[i],
                a_cpu[i]
            );
        }
    }
}
