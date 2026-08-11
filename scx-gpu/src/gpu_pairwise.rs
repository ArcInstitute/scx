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
use cudarc::driver::safe::{CudaFunction, CudaSlice, LaunchConfig};
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
/// (matching the CPU `mean_pairwise_distance`).
///
/// For the self case pass the same slice as `a` and `b` with `na == nb`: the
/// kernel then **omits the diagonal** and divides by `na²`, which is exactly
/// what the CPU `mean_pairwise_distance_self` computes (`2·Σ_{i<j} d / n²`) and
/// what `sklearn.metrics.pairwise.cosine_distances` does when X is Y.
///
/// The diagonal has to be skipped rather than assumed zero. The gram is f32
/// while the norms are f64, so a euclidean self entry is the square root of a
/// small nonzero residual — measured on the CPU at ~6.5e-4 of mean error for
/// `n = 20`, `d = 2000`, against a documented CPU/GPU bar of `atol=1e-3` per
/// perturbation. And a cosine zero-norm row normalizes to zeros, giving
/// `1 − 0 = 1` on its own diagonal, not 0.
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

    // VRAM pre-flight: the full gram is `na×nb` f32 and dominates VRAM —
    // `n_ctrl²` for the control self-distance can reach tens of GB on large
    // control groups and OOM. When it would exceed the budget, block the
    // `a`-rows so each block's gram fits, keeping the op on GPU (no silent
    // CPU fallback) instead of OOMing.
    let budget = pairwise_gram_budget(dev);
    let gram_bytes = (na as u64).saturating_mul(nb as u64).saturating_mul(4);
    if gram_bytes > budget {
        return gpu_mean_pairwise_distance_chunked(
            dev,
            handle,
            a,
            b,
            na,
            nb,
            d,
            cosine,
            std::ptr::eq(a, b) && na == nb,
            budget,
        );
    }

    // Self-distance fast path: `compute_energy_distance_gpu` calls this with
    // the SAME slice for `a` and `b` for every control and per-perturbation
    // self term (`mean_pair(&group, &group, ..)`). When so, upload +
    // normalize + squared-norm the data once and reuse the buffer for both
    // gemm operands — halving the host→device copy, the cosine normalize, and
    // the col_sqnorm launches on those calls. Numerically identical (same
    // buffers feed the gemm / reduce).
    // `na == nb` is part of the test, not redundant: a caller may legally pass
    // the same buffer with different row counts (`(a, a, 3, 5)` = the first 3
    // rows against the first 5), and aliasing there would upload and normalize
    // the wrong extent. Mirrors the CPU `pairwise_gemm_row_sums` guard.
    let is_self = std::ptr::eq(a, b) && na == nb;

    // Row-major [n × d] host slice == col-major [d × n] device buffer.
    let mut a_dev = dev.htod_copy(a)?;
    if cosine {
        gpu_harmony_l2_normalize_cols(dev, &mut a_dev, d, na)?;
    }
    let b_dev_opt = if is_self {
        None
    } else {
        let mut b_dev = dev.htod_copy(b)?;
        if cosine {
            gpu_harmony_l2_normalize_cols(dev, &mut b_dev, d, nb)?;
        }
        Some(b_dev)
    };
    let b_dev: &CudaSlice<f32> = b_dev_opt.as_ref().unwrap_or(&a_dev);

    // gram[na × nb] (col-major) = a_i · b_j — same gemm shape as
    // gpu_harmony_distances_gemm, with alpha=1 (the −2 is applied in finalize).
    let mut gram = dev.alloc_zeros::<f32>(na * nb)?;
    gpu_sgemm(
        handle,
        dev.stream(),
        &a_dev,
        b_dev,
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
    // For a self-distance a_sq == b_sq, so compute it once and alias.
    let a_sq = if cosine {
        dev.alloc_zeros::<f64>(1)?
    } else {
        col_sqnorm(dev, &a_dev, d, na)?
    };
    let b_sq_opt = if cosine || is_self {
        None
    } else {
        Some(col_sqnorm(dev, b_dev, d, nb)?)
    };
    let b_sq: &CudaSlice<f64> = b_sq_opt.as_ref().unwrap_or(&a_sq);

    let mut out = dev.alloc_zeros::<f64>(1)?;
    let module = dev.load_module_cached(PAIRWISE_DIST_PTX)?;
    let func = module
        .load_function("pairwise_dist_sum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("pairwise_dist_sum: {e}")))?;
    launch_pairwise_sum(
        dev, &func, &gram, &a_sq, b_sq, na, nb, cosine, 0, is_self, &mut out,
    )?;
    dev.synchronize()?;

    let sum = dev.dtoh_copy(&out)?[0];
    Ok(sum / (na as f64 * nb as f64))
}

/// VRAM budget (bytes) for a single pairwise gram allocation. Above this the
/// gram is chunked over `a`-row blocks.
///
/// `SCX_GPU_PAIRWISE_MAX_GRAM_BYTES` overrides the budget (used by tests to
/// force chunking on small inputs). Otherwise ~50% of free VRAM, leaving
/// headroom for the two operands (`na·d` + `nb·d` f32), the squared-norm
/// buffers, and cuBLAS workspace. If free VRAM can't be queried, returns
/// `u64::MAX` so the caller keeps the single-shot path and the allocator
/// surfaces any OOM exactly as before.
fn pairwise_gram_budget(dev: &GpuDevice) -> u64 {
    if let Ok(v) = std::env::var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES") {
        if let Ok(bytes) = v.parse::<u64>() {
            return bytes.max(4);
        }
    }
    match dev.free_memory() {
        Ok((free, _total)) => ((free as f64 * 0.5) as u64).max(4),
        Err(_) => u64::MAX,
    }
}

/// Launch `pairwise_dist_sum_kernel` for a `[na × nb]` gram, accumulating the
/// summed pairwise distance into `out` via the kernel's `atomicAdd`. Shared by
/// the single-shot and chunked paths so the launch config lives in one place.
#[allow(clippy::too_many_arguments)]
fn launch_pairwise_sum(
    dev: &GpuDevice,
    func: &CudaFunction,
    gram: &CudaSlice<f32>,
    a_sq: &CudaSlice<f64>,
    b_sq: &CudaSlice<f64>,
    na: usize,
    nb: usize,
    cosine: bool,
    a_row_offset: usize,
    skip_diagonal: bool,
    out: &mut CudaSlice<f64>,
) -> Result<(), GpuError> {
    let threads: u32 = 256;
    let total: u64 = (na as u64) * (nb as u64);
    // Grid-stride loop covers `total` regardless; cap the grid so it stays
    // within CUDA limits for very large products.
    let blocks: u32 = total.div_ceil(threads as u64).min(65_535) as u32;
    let na_i64 = na as i64;
    let nb_i64 = nb as i64;
    let mode: i32 = if cosine { 1 } else { 0 };
    let offset_i64 = a_row_offset as i64;
    let skip_i32: i32 = i32::from(skip_diagonal);
    let smem: u32 = threads * 8; // one f64 per thread
    unsafe {
        dev.stream()
            .launch_builder(func)
            .arg(gram)
            .arg(a_sq)
            .arg(b_sq)
            .arg(&na_i64)
            .arg(&nb_i64)
            .arg(&mode)
            .arg(&offset_i64)
            .arg(&skip_i32)
            .arg(out)
            .launch(LaunchConfig {
                grid_dim: (blocks.max(1), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: smem,
            })
    }
    .map(|_| ())
    .map_err(|e| GpuError::KernelLaunchFailed(format!("pairwise_dist_sum: {e}")))
}

/// Chunked variant of [`gpu_mean_pairwise_distance`] for when the full `na×nb`
/// gram would exceed `budget` bytes of VRAM. Blocks the `a`-rows so each
/// block's gram (`na_block × nb` f32) fits the budget; uploads `b` once and
/// accumulates every block's pairwise-sum into a single `out` via the kernel's
/// `atomicAdd`, then divides by `na·nb`. Numerically equivalent to the
/// single-shot path within f64-atomic summation order. `is_self` is forwarded
/// to the kernel along with each tile's global first row, so the diagonal is
/// omitted from whichever tile happens to contain it.
#[allow(clippy::too_many_arguments)]
fn gpu_mean_pairwise_distance_chunked(
    dev: &GpuDevice,
    handle: &CublasHandle,
    a: &[f32],
    b: &[f32],
    na: usize,
    nb: usize,
    d: usize,
    cosine: bool,
    is_self: bool,
    budget: u64,
) -> Result<f64, GpuError> {
    // `b` is the columns operand — upload + normalize + squared-norm once and
    // reuse across every `a`-row block.
    let mut b_dev = dev.htod_copy(b)?;
    if cosine {
        gpu_harmony_l2_normalize_cols(dev, &mut b_dev, d, nb)?;
    }
    let b_sq = if cosine {
        dev.alloc_zeros::<f64>(1)? // cosine ignores norms (kernel mode == 1)
    } else {
        col_sqnorm(dev, &b_dev, d, nb)?
    };

    // Rows of `a` per block: keep `na_block·nb·4 <= budget`, at least one row
    // (a single row's gram is `nb·4` bytes, always tiny — progress guaranteed).
    let per_row_bytes = (nb as u64).saturating_mul(4).max(1);
    let na_block = ((budget / per_row_bytes).max(1) as usize).min(na.max(1));

    let mut gram = dev.alloc_zeros::<f32>(na_block * nb)?;
    let mut out = dev.alloc_zeros::<f64>(1)?; // accumulates across blocks
    let module = dev.load_module_cached(PAIRWISE_DIST_PTX)?;
    let func = module
        .load_function("pairwise_dist_sum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("pairwise_dist_sum: {e}")))?;

    let mut i0 = 0usize;
    while i0 < na {
        let i1 = (i0 + na_block).min(na);
        let na_b = i1 - i0;
        let mut a_dev = dev.htod_copy(&a[i0 * d..i1 * d])?;
        if cosine {
            gpu_harmony_l2_normalize_cols(dev, &mut a_dev, d, na_b)?;
        }
        // gram[na_b × nb] (col-major) = a_i · b_j; beta = 0 overwrites the reused
        // buffer (the last, smaller block writes only its `na_b·nb` prefix).
        gpu_sgemm(
            handle,
            dev.stream(),
            &a_dev,
            &b_dev,
            &mut gram,
            na_b,
            nb,
            d,
            1.0,
            0.0,
            cbs::cublasOperation_t::CUBLAS_OP_T,
            cbs::cublasOperation_t::CUBLAS_OP_N,
        )?;
        let a_sq = if cosine {
            dev.alloc_zeros::<f64>(1)?
        } else {
            col_sqnorm(dev, &a_dev, d, na_b)?
        };
        // `i0` is this tile's first global a-row, which is what lets the
        // kernel find the diagonal inside a tile that does not start at 0.
        launch_pairwise_sum(
            dev, &func, &gram, &a_sq, &b_sq, na_b, nb, cosine, i0, is_self, &mut out,
        )?;
        i0 = i1;
    }
    dev.synchronize()?;

    let sum = dev.dtoh_copy(&out)?[0];
    Ok(sum / (na as f64 * nb as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random points in ~[-1, 1), no `rand` dependency.
    fn make_points(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut v = Vec::with_capacity(n * d);
        for _ in 0..n * d {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
            v.push(u * 2.0 - 1.0);
        }
        v
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * a.abs().max(1.0)
    }

    /// The VRAM-chunked path must match the single-shot path within f64
    /// atomic-summation order, for euclidean and cosine, on cross (`a != b`)
    /// inputs — including the degenerate `na_block == 1` (row-by-row) budget.
    #[test]
    fn chunked_matches_single_shot_cross() {
        let dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();
        let (na, nb, d) = (40usize, 24usize, 8usize);
        let a = make_points(na, d, 1);
        let b = make_points(nb, d, 2);

        for &cosine in &[false, true] {
            std::env::remove_var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES");
            let reference =
                gpu_mean_pairwise_distance(&dev, &handle, &a, &b, na, nb, d, cosine).unwrap();

            // Force multi-block chunking (~4 rows per block).
            std::env::set_var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES", (nb * 4 * 4).to_string());
            let chunked =
                gpu_mean_pairwise_distance(&dev, &handle, &a, &b, na, nb, d, cosine).unwrap();

            // Force row-by-row (na_block == 1).
            std::env::set_var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES", "1");
            let per_row =
                gpu_mean_pairwise_distance(&dev, &handle, &a, &b, na, nb, d, cosine).unwrap();
            std::env::remove_var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES");

            assert!(
                close(chunked, reference),
                "cosine={cosine}: chunked={chunked} vs ref={reference}"
            );
            assert!(
                close(per_row, reference),
                "cosine={cosine}: per_row={per_row} vs ref={reference}"
            );
        }
    }

    /// Chunking must also match the single-shot `is_self` fast path when
    /// `a == b` (the control self-distance term that OOMs at scale).
    #[test]
    fn chunked_matches_single_shot_self() {
        let dev = require_gpu!();
        let handle = CublasHandle::new().unwrap();
        let (n, d) = (48usize, 6usize);
        let a = make_points(n, d, 7);

        for &cosine in &[false, true] {
            std::env::remove_var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES");
            let reference =
                gpu_mean_pairwise_distance(&dev, &handle, &a, &a, n, n, d, cosine).unwrap();

            std::env::set_var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES", (n * 4 * 3).to_string());
            let chunked =
                gpu_mean_pairwise_distance(&dev, &handle, &a, &a, n, n, d, cosine).unwrap();
            std::env::remove_var("SCX_GPU_PAIRWISE_MAX_GRAM_BYTES");

            assert!(
                close(chunked, reference),
                "self cosine={cosine}: chunked={chunked} vs ref={reference}"
            );
        }
    }
}
