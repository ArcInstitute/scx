//! cuRAND wrapper for GPU-side random number generation.
//!
//! Provides [`random_gaussian_gpu`] for generating Gaussian random matrices
//! directly on GPU, avoiding host-side generation + upload for the Ω matrix
//! in randomized PCA.

use std::sync::Arc;

use cudarc::curand::result as crand;
use cudarc::curand::sys as crand_sys;
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtrMut};

use crate::device::GpuDevice;
use crate::error::GpuError;

/// RAII wrapper around a cuRAND generator (`curandGenerator_t`).
///
/// `Drop` calls `curandDestroyGenerator` so the handle is released on every
/// path — success, `?`-error, or panic — closing the leak where the previous
/// explicit destroy only ran on the success path. Function-local (not stored
/// across threads), so unlike the cuBLAS/cuSPARSE/cuSOLVER handles it needs no
/// `Send`/`Sync`.
struct CurandGenerator {
    raw: crand_sys::curandGenerator_t,
}

impl CurandGenerator {
    fn new(kind: crand_sys::curandRngType_t) -> Result<Self, GpuError> {
        let raw = crand::create_generator_kind(kind)
            .map_err(|e| GpuError::CuRandError(format!("curandCreateGenerator: {e:?}")))?;
        Ok(Self { raw })
    }

    fn raw(&self) -> crand_sys::curandGenerator_t {
        self.raw
    }
}

impl Drop for CurandGenerator {
    fn drop(&mut self) {
        // Mirror the cuBLAS/cuSPARSE/cuSOLVER handle drops: best-effort
        // destroy, error swallowed (nothing actionable in `drop`).
        unsafe {
            let _ = crand::destroy_generator(self.raw);
        }
    }
}

/// Generate a random Gaussian matrix directly on GPU.
///
/// Uses cuRAND's XORWOW generator for speed. The output is a `CudaSlice<f32>`
/// of `rows × cols` elements drawn from N(0, 1).
///
/// **Note:** cuRAND requires an even number of elements for normal generation.
/// If `rows × cols` is odd, we allocate one extra element and trim.
///
/// # Arguments
///
/// * `dev` — GPU device
/// * `stream` — CUDA stream for generator binding
/// * `rows` — Number of rows
/// * `cols` — Number of columns
/// * `seed` — Random seed for reproducibility
pub fn random_gaussian_gpu(
    dev: &GpuDevice,
    stream: &Arc<CudaStream>,
    rows: usize,
    cols: usize,
    seed: u64,
) -> Result<CudaSlice<f32>, GpuError> {
    let total = rows * cols;
    if total == 0 {
        return dev.alloc_zeros::<f32>(0);
    }

    // cuRAND requires even count for normal generation (Box-Muller).
    let alloc_count = if total % 2 == 1 { total + 1 } else { total };

    // Create generator (XORWOW — fast, adequate quality for PCA). The RAII
    // wrapper destroys it on every exit path below, including the `?`-errors.
    let gen = CurandGenerator::new(crand_sys::curandRngType_t::CURAND_RNG_PSEUDO_XORWOW)?;

    // Set seed
    unsafe {
        crand::set_seed(gen.raw(), seed)
            .map_err(|e| GpuError::CuRandError(format!("curandSetSeed: {e:?}")))?;
    }

    // Set stream
    unsafe {
        crand::set_stream(gen.raw(), stream.cu_stream() as _)
            .map_err(|e| GpuError::CuRandError(format!("curandSetStream: {e:?}")))?;
    }

    // Allocate device memory
    let mut buf = dev.alloc_zeros::<f32>(alloc_count)?;

    // Generate N(0, 1) directly on GPU
    {
        let (buf_ptr, _guard) = buf.device_ptr_mut(stream);
        unsafe {
            crand::generate::normal_f32(gen.raw(), buf_ptr as *mut f32, alloc_count, 0.0, 1.0)
                .map_err(|e| GpuError::CuRandError(format!("curandGenerateNormal: {e:?}")))?;
        }
    }

    // `gen` is destroyed by its `Drop` impl when it falls out of scope below
    // (after the optional trim) — no explicit `destroy_generator` needed.

    // If we allocated an extra element for even count, trim via round-trip.
    // Only 1 extra element so the overhead is trivial.
    if alloc_count != total {
        let mut host = dev.dtoh_copy(&buf)?;
        host.truncate(total);
        Ok(dev.htod_copy(&host)?)
    } else {
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_random_gaussian_gpu_basic() {
        let dev = require_gpu!();

        let rows = 100;
        let cols = 60;
        let buf = random_gaussian_gpu(&dev, dev.stream(), rows, cols, 42).unwrap();
        dev.synchronize().unwrap();

        let host = dev.dtoh_copy(&buf).unwrap();
        assert_eq!(host.len(), rows * cols);

        // Mean should be approximately 0 (within 3σ/√n)
        let mean: f32 = host.iter().sum::<f32>() / host.len() as f32;
        assert!(mean.abs() < 0.2, "mean = {mean}, expected approximately 0");

        // Std should be approximately 1
        let var: f32 =
            host.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / host.len() as f32;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 0.15,
            "std = {std}, expected approximately 1"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_random_gaussian_gpu_odd_count() {
        let dev = require_gpu!();

        // 7 × 3 = 21 elements (odd) — tests the padding logic
        let buf = random_gaussian_gpu(&dev, dev.stream(), 7, 3, 123).unwrap();
        dev.synchronize().unwrap();

        let host = dev.dtoh_copy(&buf).unwrap();
        assert_eq!(host.len(), 21);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_random_gaussian_gpu_reproducible() {
        let dev = require_gpu!();

        let buf1 = random_gaussian_gpu(&dev, dev.stream(), 50, 10, 42).unwrap();
        let buf2 = random_gaussian_gpu(&dev, dev.stream(), 50, 10, 42).unwrap();
        dev.synchronize().unwrap();

        let h1 = dev.dtoh_copy(&buf1).unwrap();
        let h2 = dev.dtoh_copy(&buf2).unwrap();

        // Same seed → same values
        assert_eq!(h1.len(), h2.len());
        for i in 0..h1.len() {
            assert!(
                (h1[i] - h2[i]).abs() < 1e-6,
                "reproducibility: index {i}: {} vs {}",
                h1[i],
                h2[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_random_gaussian_gpu_empty() {
        let dev = require_gpu!();

        let buf = random_gaussian_gpu(&dev, dev.stream(), 0, 0, 42).unwrap();
        assert_eq!(buf.len(), 0);
    }
}
