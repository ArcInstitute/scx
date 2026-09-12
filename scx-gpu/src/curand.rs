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
/// **Note:** cuRAND requires an even number of elements for normal generation
/// (Box-Muller produces them in pairs). If `rows × cols` is odd we generate
/// into an `alloc_count = total + 1` scratch and copy `[..total]` into an
/// exact-length buffer with one **device-to-device** `memcpy_dtod` — cudarc
/// 0.19's `CudaSlice` has no owned shrink, and a host round-trip to drop one
/// element would be both a full-buffer D2H+H2D and a `capture_guard` violation
/// (`memcpy_dtod` carries no such guard; `dtoh_copy` / `htod_copy` do).
///
/// The odd branch is **not on any default path**: the only production caller is
/// `gpu_pca::randomized_pca_core`, where `k = n_components + n_oversamples`
/// clamped by `n_vars` / `n_obs`, and every pyscx default gives `k = 60`. It
/// takes a caller passing an odd `n_comps + n_oversamples` *and* an odd clamped
/// dimension to reach it. So this is a correctness path, not a hot one — do not
/// quote a per-PCA saving for it.
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

    // If we allocated an extra element for the even count, trim on the device:
    // the surviving values are the scratch's first `total`, bit-for-bit, so the
    // random subspace — and therefore PCA's output — is unchanged from what the
    // generator produced. Taking any other window would silently change it.
    if alloc_count != total {
        let mut out = dev.alloc_zeros::<f32>(total)?;
        let src = buf.try_slice(..total).ok_or_else(|| {
            GpuError::CudaError(format!(
                "random_gaussian_gpu: could not view scratch[..{total}] of {alloc_count}"
            ))
        })?;
        stream
            .memcpy_dtod(&src, &mut out)
            .map_err(|e| GpuError::CudaError(format!("dtod trim (random_gaussian_gpu): {e}")))?;
        // `buf` — the scratch `src` views — is dropped when this block ends, and
        // cudarc frees it stream-ordered on the stream it was ALLOCATED on
        // (`dev`'s), not necessarily the `stream` the copy was queued on. Every
        // caller today passes `dev.stream()`, so the two coincide; a caller that
        // did not would free the source out from under an in-flight copy. Sync
        // rather than rely on an invariant the signature does not enforce.
        //
        // This is the one place the D2D form reintroduces a host block, and it
        // is free in practice: the branch is unreachable at every pyscx default
        // (`k = 60`, so `n_vars * k` is even). It also makes this path
        // capture-illegal again — `GpuDevice::synchronize` is `capture_guard`-
        // checked, which is the correct outcome, since a host sync inside a
        // capture region is exactly what that guard exists to reject.
        dev.synchronize()?;
        Ok(out)
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

    /// The odd-count trim keeps the generator's **first** `total` values,
    /// bit-for-bit.
    ///
    /// Until the device trim landed this test asserted `host.len() == 21` and
    /// nothing else, which cannot distinguish the right 21 of 22 elements from
    /// any other 21 — and picking a different window silently changes the
    /// random subspace, and therefore PCA's output, without failing anything.
    ///
    /// The oracle is cuRAND itself: `alloc_count` is `total + 1`, the generator
    /// is seeded identically, and XORWOW is deterministic, so an *even* request
    /// for `total + 1` elements at the same seed produces exactly the scratch
    /// this odd request generates internally. Its first `total` must be what
    /// comes back.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_random_gaussian_gpu_odd_count() {
        let dev = require_gpu!();

        // 7 × 3 = 21 elements (odd) — takes the pad-and-trim branch.
        let buf = random_gaussian_gpu(&dev, dev.stream(), 7, 3, 123).unwrap();
        dev.synchronize().unwrap();
        let host = dev.dtoh_copy(&buf).unwrap();
        assert_eq!(host.len(), 21);

        // 1 × 22 = 22 elements (even) — the same generator, the same seed, no
        // trim. This is the untrimmed scratch the odd call allocated.
        let scratch = random_gaussian_gpu(&dev, dev.stream(), 1, 22, 123).unwrap();
        dev.synchronize().unwrap();
        let scratch_host = dev.dtoh_copy(&scratch).unwrap();
        assert_eq!(scratch_host.len(), 22);

        for (i, (&got, &want)) in host.iter().zip(scratch_host.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "trimmed element {i}: {got} != scratch prefix {want}"
            );
        }
    }

    /// The odd-count path is reproducible across calls, like the even one.
    ///
    /// `test_random_gaussian_gpu_reproducible` covers 500 elements (even), so
    /// before this the trim branch had no same-seed-same-values coverage at all.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_random_gaussian_gpu_odd_count_reproducible() {
        let dev = require_gpu!();

        let a = random_gaussian_gpu(&dev, dev.stream(), 5, 9, 7).unwrap();
        let b = random_gaussian_gpu(&dev, dev.stream(), 5, 9, 7).unwrap();
        dev.synchronize().unwrap();

        let ha = dev.dtoh_copy(&a).unwrap();
        let hb = dev.dtoh_copy(&b).unwrap();
        assert_eq!(ha.len(), 45);
        assert_eq!(hb.len(), 45);
        for (i, (&x, &y)) in ha.iter().zip(hb.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i} differs across calls");
        }
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
