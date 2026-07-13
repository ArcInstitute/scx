//! GPU pairwise-distance mean, backing GPU `energy_distance` (Phase 3 of
//! CELL-EVAL-SCX-GPU-ACC).
//!
//! Computes `mean over (i,j) of d(a_i, b_j)` for two point sets `a` `[na × d]`
//! and `b` `[nb × d]` (row-major host slices), via the
//! `‖x−y‖² = ‖x‖² + ‖y‖² − 2·x·yᵀ` decomposition: one cuBLAS gemm for the
//! `x·yᵀ` gram, per-point squared norms, and a finalize+reduce kernel that sums
//! the distances in `double`. Reuses `gpu_sgemm` (`cublas.rs`), the gram-shape
//! of `gpu_harmony_distances_gemm`, and `gpu_harmony_l2_normalize_cols` (cosine
//! pre-normalization). Only euclidean + cosine are gemm-decomposable; L1 has no
//! decomposition and stays on the CPU (the caller routes it there).
//!
//! Precision: the gemm runs in f32 (matching the CPU `DistanceBackend::Gemm`
//! path, which also runs the matmul at f32); squared norms and the sum
//! accumulate in f64. Parity with the CPU is at the `atol=1e-4` bar.

use cudarc::cublas::sys as cbs;
use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::cublas::{gpu_sgemm, CublasHandle};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_harmony::gpu_harmony_l2_normalize_cols;

const PAIRWISE_DIST_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pairwise_dist.ptx"));

/// Per-column (per-point) squared L2 norm of a col-major `[d × n]` device
/// matrix, accumulated in f64.
fn col_sqnorm(
    dev: &GpuDevice,
    m: &CudaSlice<f32>,
    d: usize,
    n: usize,
) -> Result<CudaSlice<f64>, GpuError> {
    let mut out = dev.alloc_zeros::<f64>(n.max(1))?;
    if n == 0 || d == 0 {
        return Ok(out);
    }
    let module = dev.load_module_cached(PAIRWISE_DIST_PTX)?;
    let func = module
        .load_function("col_sqnorm_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sqnorm: {e}")))?;
    let threads: u32 = 256;
    let blocks: u32 = (n as u32).div_ceil(threads);
    let d_i32 = d as i32;
    let n_i32 = n as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(m)
            .arg(&mut out)
            .arg(&d_i32)
            .arg(&n_i32)
            .launch(LaunchConfig {
                grid_dim: (blocks.max(1), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sqnorm: {e}")))?;
    Ok(out)
}

/// Mean of all `na·nb` pairwise distances between row-major point sets
/// `a` `[na × d]` and `b` `[nb × d]` (f32 host slices). `cosine=false` →
/// euclidean; `cosine=true` → cosine distance (`1 − cosine_sim`, clamped
/// `[0,2]`, inputs L2-normalized first). Returns `0.0` when either set is empty
/// (matching the CPU `mean_pairwise_distance`). For the self case pass the same
/// slice as `a` and `b` — the diagonal contributes 0 (clamped) and the mean is
/// over `na²`, matching `mean_pairwise_distance_self`.
#[allow(clippy::too_many_arguments)]
pub fn gpu_mean_pairwise_distance(
    dev: &GpuDevice,
    handle: &CublasHandle,
    a: &[f32],
    b: &[f32],
    na: usize,
    nb: usize,
    d: usize,
    cosine: bool,
) -> Result<f64, GpuError> {
    if na == 0 || nb == 0 {
        return Ok(0.0);
    }
    if a.len() < na * d || b.len() < nb * d {
        return Err(GpuError::ShapeMismatch {
            expected: format!("a >= na*d = {}, b >= nb*d = {}", na * d, nb * d),
            got: format!("a={}, b={}", a.len(), b.len()),
        });
    }

    // Row-major [n × d] host slice == col-major [d × n] device buffer.
    let mut a_dev = dev.htod_copy(a)?;
    let mut b_dev = dev.htod_copy(b)?;

    if cosine {
        gpu_harmony_l2_normalize_cols(dev, &mut a_dev, d, na)?;
        gpu_harmony_l2_normalize_cols(dev, &mut b_dev, d, nb)?;
    }

    // gram[na × nb] (col-major) = a_i · b_j — same gemm shape as
    // gpu_harmony_distances_gemm, with alpha=1 (the −2 is applied in finalize).
    let mut gram = dev.alloc_zeros::<f32>(na * nb)?;
    gpu_sgemm(
        handle,
        dev.stream(),
        &a_dev,
        &b_dev,
        &mut gram,
        na,
        nb,
        d,
        1.0,
        0.0,
        cbs::cublasOperation_t::CUBLAS_OP_T,
        cbs::cublasOperation_t::CUBLAS_OP_N,
    )?;

    // Squared norms (euclidean only; cosine ignores them — pass dummies).
    let (a_sq, b_sq) = if cosine {
        (dev.alloc_zeros::<f64>(1)?, dev.alloc_zeros::<f64>(1)?)
    } else {
        (
            col_sqnorm(dev, &a_dev, d, na)?,
            col_sqnorm(dev, &b_dev, d, nb)?,
        )
    };

    let mut out = dev.alloc_zeros::<f64>(1)?;
    let module = dev.load_module_cached(PAIRWISE_DIST_PTX)?;
    let func = module
        .load_function("pairwise_dist_sum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("pairwise_dist_sum: {e}")))?;
    let threads: u32 = 256;
    let total: u64 = (na as u64) * (nb as u64);
    // Grid-stride loop covers `total` regardless; cap the grid so it stays
    // within CUDA limits for very large products.
    let blocks: u32 = total.div_ceil(threads as u64).min(65_535) as u32;
    let na_i64 = na as i64;
    let nb_i64 = nb as i64;
    let mode: i32 = if cosine { 1 } else { 0 };
    let smem: u32 = threads * 8; // one f64 per thread
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(&gram)
            .arg(&a_sq)
            .arg(&b_sq)
            .arg(&na_i64)
            .arg(&nb_i64)
            .arg(&mode)
            .arg(&mut out)
            .launch(LaunchConfig {
                grid_dim: (blocks.max(1), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: smem,
            })
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("pairwise_dist_sum: {e}")))?;
    dev.synchronize()?;

    let sum = dev.dtoh_copy(&out)?[0];
    Ok(sum / (na as f64 * nb as f64))
}
