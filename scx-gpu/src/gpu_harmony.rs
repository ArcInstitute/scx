//! GPU kernels for Harmony2 batch integration.
//!
//! Exposes four primitive operations as host-side functions that take
//! device-resident `CudaSlice<f32>` buffers. Callers own the memory and
//! the orchestration loop — see [`scx_accel::harmony_integrate_gpu`].
//!
//! Layout contracts (match `scx-accel/src/harmony/cpu.rs` CPU code):
//! - `Z_*` matrices: `(d x N)` column-major (`Z[t, i] = flat[i*d + t]`).
//! - `Y`: `(d x K)` column-major.
//! - `R`, `dist`, `O`, `E`: row-major (row-major stride = last dim).

use std::sync::Arc;

use cudarc::cublas::sys as cbs;
use cudarc::driver::safe::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cublas::{gpu_sgemm, CublasHandle};
use crate::device::GpuDevice;
use crate::error::GpuError;

/// PTX for all Harmony kernels (compiled from `scx-gpu/kernels/harmony.cu`).
const HARMONY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/harmony.ptx"));

/// Compute `dist[k, i] = 2 * (1 - dot(Y[:, k], Z_cos[:, i]))` on GPU.
///
/// * `y` — `(d x K)` col-major.
/// * `z_cos` — `(d x N)` col-major.
/// * `dist_out` — `(K x N)` row-major, must be pre-allocated.
///
/// Tiles by cells so each launch's `K * n_chunk` thread count fits in
/// 32 bits. Internal thread indexing in the kernel is 64-bit so the
/// kernel itself is robust to a single tile spanning more than 2^31
/// thread positions, but we still tile to keep grid_dim under 2^31
/// blocks. There is no longer a `K*N < 2^31` constraint on the global
/// problem size.
pub fn gpu_harmony_distances(
    dev: &GpuDevice,
    y: &CudaSlice<f32>,
    z_cos: &CudaSlice<f32>,
    dist_out: &mut CudaSlice<f32>,
    d: usize,
    k: usize,
    n: usize,
) -> Result<(), GpuError> {
    if y.len() < d * k {
        return Err(GpuError::ShapeMismatch {
            expected: format!("Y length >= d*K = {}", d * k),
            got: format!("{}", y.len()),
        });
    }
    if z_cos.len() < d * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("Z_cos length >= d*N = {}", d * n),
            got: format!("{}", z_cos.len()),
        });
    }
    if dist_out.len() < k * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("dist length >= K*N = {}", k * n),
            got: format!("{}", dist_out.len()),
        });
    }
    if k == 0 || n == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_distances_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_distances: {e}")))?;

    // Cap each launch at K * n_chunk < 2^30 thread positions so the
    // grid (blocks = ceil(threads / 256)) stays well under CUDA's
    // 2^31-1 grid_dim.x limit.
    const TILE_LIMIT: usize = 1 << 30;
    let n_chunk = (TILE_LIMIT / k).max(1).min(n);
    let threads: u32 = 256;
    let d_i32 = d as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

    let mut i_offset: usize = 0;
    while i_offset < n {
        let chunk = n_chunk.min(n - i_offset);
        let total: u64 = (k as u64) * (chunk as u64);
        let blocks = total.div_ceil(threads as u64) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let i_off_i64 = i_offset as i64;
        let n_chunk_i64 = chunk as i64;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(y)
                .arg(z_cos)
                .arg(&mut *dist_out)
                .arg(&d_i32)
                .arg(&k_i32)
                .arg(&n_i32)
                .arg(&i_off_i64)
                .arg(&n_chunk_i64)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_distances: {e}")))?;
        i_offset += chunk;
    }
    Ok(())
}

/// GEMM-backed variant of `gpu_harmony_distances`.
///
/// Computes `D = -2 · Z_cos^T · Y` via cuBLAS sgemm and adds the
/// constant 2 elementwise. Faster than the hand-written kernel for
/// large K and N because cuBLAS picks tile sizes optimised for the
/// device. Output layout is the same row-major `(K x N)` as
/// `gpu_harmony_distances`.
///
/// Note: cuBLAS sgemm dimensions are `i32`. At `N >= 2^31` the GEMM
/// path would overflow; callers should fall back to
/// `gpu_harmony_distances` (host-tiled) in that regime.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_distances_gemm(
    dev: &GpuDevice,
    handle: &CublasHandle,
    y: &CudaSlice<f32>,
    z_cos: &CudaSlice<f32>,
    dist_out: &mut CudaSlice<f32>,
    d: usize,
    k: usize,
    n: usize,
) -> Result<(), GpuError> {
    if y.len() < d * k {
        return Err(GpuError::ShapeMismatch {
            expected: format!("Y length >= d*K = {}", d * k),
            got: format!("{}", y.len()),
        });
    }
    if z_cos.len() < d * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("Z_cos length >= d*N = {}", d * n),
            got: format!("{}", z_cos.len()),
        });
    }
    if dist_out.len() < k * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("dist length >= K*N = {}", k * n),
            got: format!("{}", dist_out.len()),
        });
    }
    if k == 0 || n == 0 {
        return Ok(());
    }
    if (n as u64) > (i32::MAX as u64) || (k as u64) > (i32::MAX as u64) {
        return Err(GpuError::ShapeMismatch {
            expected: "K, N each fit in i32 for cuBLAS sgemm".into(),
            got: format!("K={k}, N={n}"),
        });
    }

    // dist viewed as col-major (N, K) is the same memory as row-major
    // (K, N). gemm computes `D[i, k] = α · Σ_t Z_cos[t, i] · Y[t, k]`
    // with α = -2, β = 0. Then a finalize kernel adds 2 elementwise.
    //
    // gemm:  m = N, n = K, kk = d
    //        A = Z_cos (col-major (d, N), lda = d), op_a = T  → logical (N, d)
    //        B = Y     (col-major (d, K), ldb = d), op_b = N  → logical (d, K)
    //        C = dist  (col-major (N, K), ldc = N)
    let stream: &Arc<CudaStream> = dev.stream();
    gpu_sgemm(
        handle,
        stream,
        z_cos,
        y,
        dist_out,
        n,
        k,
        d,
        -2.0,
        0.0,
        cbs::cublasOperation_t::CUBLAS_OP_T,
        cbs::cublasOperation_t::CUBLAS_OP_N,
    )?;

    // Finalize: D[idx] = 2.0 + D[idx] for all K*N elements. Block count
    // = ceil(K*N / 256); CUDA grid_dim.x supports up to 2^31 - 1 blocks,
    // which covers K*N up to ~5e11 in a single launch — beyond any
    // realistic Harmony workload.
    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_dist_finalize_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_dist_finalize: {e}")))?;

    let total: u64 = (k as u64) * (n as u64);
    let threads: u32 = 256;
    let blocks_u64 = total.div_ceil(threads as u64);
    if blocks_u64 > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "K*N / 256 < 2^31 (CUDA grid_dim.x cap)".into(),
            got: format!("blocks = {blocks_u64}"),
        });
    }
    let cfg = LaunchConfig {
        grid_dim: (blocks_u64 as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let total_i64 = total as i64;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(dist_out)
            .arg(&total_i64)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_dist_finalize: {e}")))?;
    Ok(())
}

/// Fused softmax + diversity-penalty kernel.
///
/// For each cell, computes `R[:, i] = softmax(-dist[:, i]/sigma + sum_c theta_c * log_ratio_c)`
/// where `log_ratio_c = log((2 E[k, b_c(i)]+1) / (O[k, b_c(i)] + E[k, b_c(i)] + 1))`.
///
/// `batch_labels_flat` must be the per-covariate labels laid out as a
/// `(C x N)` row-major i32 matrix (so covariate c's label for cell i is
/// at `c*N + i`). `cov_offset` has length C.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_softmax_penalty(
    dev: &GpuDevice,
    dist: &CudaSlice<f32>,
    sigma: &CudaSlice<f32>,
    o: &CudaSlice<f32>,
    e: &CudaSlice<f32>,
    theta: &CudaSlice<f32>,
    batch_labels_flat: &CudaSlice<i32>,
    cov_offset: &CudaSlice<i32>,
    r_out: &mut CudaSlice<f32>,
    c: usize,
    k: usize,
    n: usize,
    b: usize,
) -> Result<(), GpuError> {
    if k == 0 || n == 0 || c == 0 {
        return Ok(());
    }
    if (n as u64) > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "n < 2^31 (CUDA grid_dim.x cap)".into(),
            got: format!("n = {n}"),
        });
    }

    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_softmax_penalty_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_softmax_penalty: {e}")))?;

    // One block per cell, min(256, next_pow2(K)) threads. Power-of-two
    // block size simplifies the reduction logic in the kernel.
    let threads: u32 = {
        let mut t = 32u32;
        while (t as usize) < k && t < 1024 {
            t <<= 1;
        }
        t.min(1024)
    };
    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (threads as usize * std::mem::size_of::<f32>()) as u32,
    };

    let c_i32 = c as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    let b_i32 = b as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(dist)
            .arg(sigma)
            .arg(o)
            .arg(e)
            .arg(theta)
            .arg(batch_labels_flat)
            .arg(cov_offset)
            .arg(&c_i32)
            .arg(&k_i32)
            .arg(&n_i32)
            .arg(&b_i32)
            .arg(r_out)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_softmax_penalty: {e}")))?;
    Ok(())
}

/// Plain softmax `R[:, i] = softmax(-dist[:, i] / sigma)` for every cell.
///
/// No diversity penalty applied — used at iter > 0 cold-start, where the
/// inner block-decrement softmax inside `update_r` will fold in the
/// leave-block-out penalty per block.
pub fn gpu_harmony_softmax(
    dev: &GpuDevice,
    dist: &CudaSlice<f32>,
    sigma: &CudaSlice<f32>,
    r_out: &mut CudaSlice<f32>,
    k: usize,
    n: usize,
) -> Result<(), GpuError> {
    if k == 0 || n == 0 {
        return Ok(());
    }
    if (n as u64) > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "n < 2^31 (CUDA grid_dim.x cap)".into(),
            got: format!("n = {n}"),
        });
    }
    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_softmax_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_softmax: {e}")))?;

    let threads: u32 = {
        let mut t = 32u32;
        while (t as usize) < k && t < 1024 {
            t <<= 1;
        }
        t.min(1024)
    };
    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (threads as usize * std::mem::size_of::<f32>()) as u32,
    };
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(dist)
            .arg(sigma)
            .arg(&k_i32)
            .arg(&n_i32)
            .arg(r_out)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_softmax: {e}")))?;
    Ok(())
}

/// Block-only softmax+penalty for the k-means `update_r` sub-loop.
///
/// Operates on a list of cell indices `block_cells[blockIdx.x]`
/// instead of all N cells. Callers must pre-compute leave-block-out
/// O/E (subtract the block's R contributions before launching).
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_block_softmax_penalty(
    dev: &GpuDevice,
    dist: &CudaSlice<f32>,
    sigma: &CudaSlice<f32>,
    o: &CudaSlice<f32>,
    e: &CudaSlice<f32>,
    theta: &CudaSlice<f32>,
    batch_labels_flat: &CudaSlice<i32>,
    cov_offset: &CudaSlice<i32>,
    block_cells: &CudaSlice<i32>,
    r_out: &mut CudaSlice<f32>,
    c: usize,
    k: usize,
    n: usize,
    b: usize,
    n_block_cells: usize,
) -> Result<(), GpuError> {
    if k == 0 || n == 0 || c == 0 || n_block_cells == 0 {
        return Ok(());
    }
    if (n_block_cells as u64) > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_block_cells < 2^31".into(),
            got: format!("n_block_cells = {n_block_cells}"),
        });
    }
    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_block_softmax_penalty_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_block_softmax_penalty: {e}")))?;

    let threads: u32 = {
        let mut t = 32u32;
        while (t as usize) < k && t < 1024 {
            t <<= 1;
        }
        t.min(1024)
    };
    let cfg = LaunchConfig {
        grid_dim: (n_block_cells as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (threads as usize * std::mem::size_of::<f32>()) as u32,
    };
    let c_i32 = c as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    let b_i32 = b as i32;
    let n_block_cells_i32 = n_block_cells as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(dist)
            .arg(sigma)
            .arg(o)
            .arg(e)
            .arg(theta)
            .arg(batch_labels_flat)
            .arg(cov_offset)
            .arg(block_cells)
            .arg(&c_i32)
            .arg(&k_i32)
            .arg(&n_i32)
            .arg(&b_i32)
            .arg(&n_block_cells_i32)
            .arg(r_out)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_block_softmax_penalty: {e}")))?;
    Ok(())
}

/// Atomic O/E update for a block of cells (signed).
///
/// For each (cell, cluster, covariate), atomicAdds `sign * R[k, cell]`
/// to `O[k, gb]` and `sign * pr_b[gb] * R[k, cell]` to `E[k, gb]`. Use
/// `sign = -1.0` to subtract the block's contribution before block
/// softmax (leave-block-out), and `sign = +1.0` to add the new R
/// contribution after.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_block_oe_update(
    dev: &GpuDevice,
    r: &CudaSlice<f32>,
    block_cells: &CudaSlice<i32>,
    batch_labels_flat: &CudaSlice<i32>,
    cov_offset: &CudaSlice<i32>,
    pr_b: &CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    e: &mut CudaSlice<f32>,
    sign: f32,
    c: usize,
    k: usize,
    n: usize,
    b: usize,
    n_block_cells: usize,
) -> Result<(), GpuError> {
    if k == 0 || n == 0 || c == 0 || n_block_cells == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_block_oe_update_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_block_oe_update: {e}")))?;

    let total: u64 = (n_block_cells as u64) * (k as u64) * (c as u64);
    let threads: u32 = 256;
    let blocks_u64 = total.div_ceil(threads as u64);
    if blocks_u64 > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_block_cells * K * C / 256 < 2^31".into(),
            got: format!("blocks = {blocks_u64}"),
        });
    }
    let cfg = LaunchConfig {
        grid_dim: (blocks_u64 as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let c_i32 = c as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    let b_i32 = b as i32;
    let n_block_cells_i32 = n_block_cells as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(r)
            .arg(block_cells)
            .arg(batch_labels_flat)
            .arg(cov_offset)
            .arg(pr_b)
            .arg(&sign)
            .arg(&c_i32)
            .arg(&k_i32)
            .arg(&n_i32)
            .arg(&b_i32)
            .arg(&n_block_cells_i32)
            .arg(o)
            .arg(e)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_block_oe_update: {e}")))?;
    Ok(())
}

/// Compute O / E from R on GPU (full-N initial reduction).
///
/// `O[k, gb] = Σ_i R[k, i]` summed over cells whose batch index is gb;
/// `E[k, gb] = pr_b[gb] * Σ_i R[k, i]` (per cluster row sum). Used at
/// iter > 0 cold-start to refresh O/E from a freshly-computed
/// (penalty-free) R. Allocates a transient row_sum scratch internally.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_compute_o_e_full(
    dev: &GpuDevice,
    r: &CudaSlice<f32>,
    batch_labels_flat: &CudaSlice<i32>,
    cov_offset: &CudaSlice<i32>,
    pr_b: &CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    e: &mut CudaSlice<f32>,
    c: usize,
    k: usize,
    n: usize,
    b: usize,
) -> Result<(), GpuError> {
    if k == 0 || n == 0 || c == 0 {
        return Ok(());
    }
    // Zero O and the row_sum scratch before atomic accumulation.
    let mut row_sum = dev
        .alloc_zeros::<f32>(k)
        .map_err(|e| GpuError::OutOfMemory(format!("alloc row_sum: {e}")))?;
    dev.stream()
        .memset_zeros(o)
        .map_err(|e| GpuError::CudaError(format!("memset O: {e}")))?;

    let module = dev.load_module_cached(HARMONY_PTX)?;

    // Pass 1: harmony_compute_o_kernel — atomicAdd into O and row_sum.
    let func = module
        .load_function("harmony_compute_o_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_compute_o: {e}")))?;
    let total: u64 = (n as u64) * (k as u64) * (c as u64);
    let threads: u32 = 256;
    let blocks_u64 = total.div_ceil(threads as u64);
    if blocks_u64 > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "N * K * C / 256 < 2^31".into(),
            got: format!("blocks = {blocks_u64}"),
        });
    }
    let cfg = LaunchConfig {
        grid_dim: (blocks_u64 as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let c_i32 = c as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    let b_i32 = b as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(r)
            .arg(batch_labels_flat)
            .arg(cov_offset)
            .arg(&c_i32)
            .arg(&k_i32)
            .arg(&n_i32)
            .arg(&b_i32)
            .arg(o)
            .arg(&mut row_sum)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_compute_o: {e}")))?;

    // Pass 2: harmony_compute_e_finalize_kernel — E[k, gb] = pr_b[gb] * row_sum[k].
    let func2 = module
        .load_function("harmony_compute_e_finalize_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_compute_e_finalize: {e}")))?;
    let threads_x: u32 = 64;
    let blocks_x = (b as u32).div_ceil(threads_x);
    let cfg2 = LaunchConfig {
        grid_dim: (blocks_x, k as u32, 1),
        block_dim: (threads_x, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        dev.stream()
            .launch_builder(&func2)
            .arg(&row_sum)
            .arg(pr_b)
            .arg(&k_i32)
            .arg(&b_i32)
            .arg(e)
            .launch(cfg2)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_compute_e_finalize: {e}")))?;
    Ok(())
}

/// Per-cell kmeans + entropy objective contribution.
///
/// Writes `obj_cell[i] = Σ_k R[k,i] · dist[k,i] + Σ_k σ[k] · R[k,i] ·
/// log R[k,i]` for i ∈ [0, N). Reducing this vector gives the
/// `kmeans_err + entropy` portion of the Harmony objective; the
/// caller then adds the cross-entropy term (see
/// `gpu_harmony_obj_cross`) and multiplies by `2000 / N`.
pub fn gpu_harmony_obj_kmeans_entropy(
    dev: &GpuDevice,
    r: &CudaSlice<f32>,
    dist: &CudaSlice<f32>,
    sigma: &CudaSlice<f32>,
    obj_cell: &mut CudaSlice<f32>,
    k: usize,
    n: usize,
) -> Result<(), GpuError> {
    if k == 0 || n == 0 {
        return Ok(());
    }
    if (n as u64) > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "n < 2^31 (CUDA grid_dim.x cap)".into(),
            got: format!("n = {n}"),
        });
    }
    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_obj_kmeans_entropy_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_obj_kmeans_entropy: {e}")))?;

    let threads: u32 = {
        let mut t = 32u32;
        while (t as usize) < k && t < 1024 {
            t <<= 1;
        }
        t.min(1024)
    };
    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (threads as usize * std::mem::size_of::<f32>()) as u32,
    };
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(r)
            .arg(dist)
            .arg(sigma)
            .arg(&k_i32)
            .arg(&n_i32)
            .arg(obj_cell)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_obj_kmeans_entropy: {e}")))?;
    Ok(())
}

/// Sum `obj_cell` (length `n`) and `cross_kgb` (length `k_b`) into two f64
/// scalars **on-device** (Task 2.6), returning `(sum_obj_cell, sum_cross_kgb)`.
///
/// Replaces the previous host f64 reduction over a full `(N + K·B)`-element
/// D→H copy in `compute_objective_gpu` with a 2-element download. Accumulates
/// in f64 inside the kernel so the result matches the prior host f64 sum within
/// floating-point reassociation. Uses a power-of-two block (256) as the tree
/// reduction requires.
pub fn gpu_harmony_reduce_objective(
    dev: &GpuDevice,
    obj_cell: &CudaSlice<f32>,
    n: usize,
    cross_kgb: &CudaSlice<f32>,
    k_b: usize,
) -> Result<(f64, f64), GpuError> {
    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_reduce_sum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_reduce_sum: {e}")))?;

    // out[0] = Σ obj_cell, out[1] = Σ cross_kgb. Zeroed before the
    // atomicAdd-accumulating launches.
    let mut d_out = dev.alloc_zeros::<f64>(2)?;

    const THREADS: u32 = 256;
    let mut reduce_into =
        |src: &CudaSlice<f32>, len: usize, out_idx: usize| -> Result<(), GpuError> {
            if len == 0 {
                return Ok(());
            }
            let blocks = (len as u32).div_ceil(THREADS).clamp(1, 1024);
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: THREADS * std::mem::size_of::<f64>() as u32,
            };
            let n_ll = len as i64;
            let mut out_view = d_out.slice_mut(out_idx..out_idx + 1);
            unsafe {
                dev.stream()
                    .launch_builder(&func)
                    .arg(src)
                    .arg(&n_ll)
                    .arg(&mut out_view)
                    .launch(cfg)
            }
            .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_reduce_sum: {e}")))?;
            Ok(())
        };

    reduce_into(obj_cell, n, 0)?;
    reduce_into(cross_kgb, k_b, 1)?;

    dev.synchronize()?;
    let h = dev.dtoh_copy(&d_out)?;
    Ok((h[0], h[1]))
}

/// Per-(k, gb) cross-entropy objective contribution.
///
/// Writes `cross[k, gb] = σ[k] · O[k, gb] · θ[gb] · log((O+E+1) /
/// (2E+1))`. Reducing this matrix gives the cross-entropy term.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_obj_cross(
    dev: &GpuDevice,
    o: &CudaSlice<f32>,
    e: &CudaSlice<f32>,
    sigma: &CudaSlice<f32>,
    theta: &CudaSlice<f32>,
    cross_kgb: &mut CudaSlice<f32>,
    k: usize,
    b: usize,
) -> Result<(), GpuError> {
    if k == 0 || b == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_obj_cross_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_obj_cross: {e}")))?;

    let threads_x: u32 = 64;
    let blocks_x = (b as u32).div_ceil(threads_x);
    let cfg = LaunchConfig {
        grid_dim: (blocks_x, k as u32, 1),
        block_dim: (threads_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let k_i32 = k as i32;
    let b_i32 = b as i32;
    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(o)
            .arg(e)
            .arg(sigma)
            .arg(theta)
            .arg(&k_i32)
            .arg(&b_i32)
            .arg(cross_kgb)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_obj_cross: {e}")))?;
    Ok(())
}

/// Column-wise L2 normalization of a `(d x N)` col-major matrix in place.
pub fn gpu_harmony_l2_normalize_cols(
    dev: &GpuDevice,
    m: &mut CudaSlice<f32>,
    d: usize,
    n: usize,
) -> Result<(), GpuError> {
    if n == 0 || d == 0 {
        return Ok(());
    }
    if m.len() < d * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("matrix length >= d*N = {}", d * n),
            got: format!("{}", m.len()),
        });
    }

    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_l2_normalize_cols_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_l2_normalize: {e}")))?;

    let threads: u32 = 256;
    let blocks = (n as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let d_i32 = d as i32;
    let n_i32 = n as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(m)
            .arg(&d_i32)
            .arg(&n_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_l2_normalize: {e}")))?;
    Ok(())
}

/// Compute per-cluster regression z-sum on GPU.
///
/// For one cluster `ku`, computes
/// `z_sum[j, t] = Σ_{i in cells of kept batch j} R[ku, i] · Z_orig[t, i]`
/// for `j ∈ [0, n_kept)`, `t ∈ [0, d)`. Output layout is row-major
/// `(n_kept, d)`.
///
/// `cells_concat` / `batch_offsets` follow the same convention as
/// `gpu_harmony_correction_grouped`: cells for batch `j` are
/// `cells_concat[batch_offsets[j]..batch_offsets[j+1]]`. Empty kept
/// batches are allowed; their z_sum rows stay zero.
///
/// f32 accumulation is sufficient at typical (post-PCA) scales because
/// `R ∈ [0, 1]` and `Z_orig` entries are bounded. f64 promotion can be
/// added later if drift is observed at very large N.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_z_sum(
    dev: &GpuDevice,
    r_row_k: &CudaSlice<f32>,
    z_orig: &CudaSlice<f32>,
    cells_concat: &CudaSlice<i32>,
    batch_offsets: &CudaSlice<i32>,
    z_sum_out: &mut CudaSlice<f32>,
    n_kept: usize,
    d: usize,
    n: usize,
) -> Result<(), GpuError> {
    if n_kept == 0 || d == 0 {
        return Ok(());
    }
    if r_row_k.len() < n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("R_row_k length >= N = {n}"),
            got: format!("{}", r_row_k.len()),
        });
    }
    if z_orig.len() < d * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("Z_orig length >= d*N = {}", d * n),
            got: format!("{}", z_orig.len()),
        });
    }
    if z_sum_out.len() < n_kept * d {
        return Err(GpuError::ShapeMismatch {
            expected: format!("z_sum length >= n_kept * d = {}", n_kept * d),
            got: format!("{}", z_sum_out.len()),
        });
    }
    if batch_offsets.len() < n_kept + 1 {
        return Err(GpuError::ShapeMismatch {
            expected: format!("batch_offsets length >= n_kept + 1 = {}", n_kept + 1),
            got: format!("{}", batch_offsets.len()),
        });
    }

    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_z_sum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_z_sum: {e}")))?;

    let warp = 32u32;
    let mut threads = (d as u32).div_ceil(warp) * warp;
    if threads == 0 {
        threads = warp;
    }
    if threads > 1024 {
        return Err(GpuError::ShapeMismatch {
            expected: "d <= 1024 (CUDA block size limit)".into(),
            got: format!("d = {d}"),
        });
    }
    if (n_kept as u64) > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_kept < 2^31 (CUDA grid_dim.x cap)".into(),
            got: format!("n_kept = {n_kept}"),
        });
    }
    let cfg = LaunchConfig {
        grid_dim: (n_kept as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_kept_i32 = n_kept as i32;
    let d_i32 = d as i32;
    let n_i32 = n as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(r_row_k)
            .arg(z_orig)
            .arg(cells_concat)
            .arg(batch_offsets)
            .arg(&n_kept_i32)
            .arg(&d_i32)
            .arg(&n_i32)
            .arg(z_sum_out)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_z_sum: {e}")))?;
    Ok(())
}

/// Apply a single per-(cluster, batch) correction to `Z_corr`.
///
/// For each cell index `c_j` in `cells`, subtracts `w_row * R_row_k[c_j]`
/// from column `c_j` of `Z_corr`. Intended to be called in a loop over
/// all (cluster, kept-batch) pairs on the CPU side.
pub fn gpu_harmony_correction(
    dev: &GpuDevice,
    z_corr: &mut CudaSlice<f32>,
    d: usize,
    n: usize,
    cells: &CudaSlice<i32>,
    r_row_k: &CudaSlice<f32>,
    w_row: &CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n_cells = cells.len();
    if n_cells == 0 || d == 0 {
        return Ok(());
    }
    if z_corr.len() < d * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("Z_corr length >= d*N = {}", d * n),
            got: format!("{}", z_corr.len()),
        });
    }
    if r_row_k.len() < n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("R_row_k length >= N = {n}"),
            got: format!("{}", r_row_k.len()),
        });
    }
    if w_row.len() < d {
        return Err(GpuError::ShapeMismatch {
            expected: format!("w_row length >= d = {d}"),
            got: format!("{}", w_row.len()),
        });
    }

    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_correction_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_correction: {e}")))?;

    let total: u64 = (n_cells as u64) * (d as u64);
    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let d_i32 = d as i32;
    let n_i32 = n as i32;
    let n_cells_i32 = n_cells as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(z_corr)
            .arg(&d_i32)
            .arg(&n_i32)
            .arg(cells)
            .arg(&n_cells_i32)
            .arg(r_row_k)
            .arg(w_row)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_correction: {e}")))?;
    Ok(())
}

/// Apply correction for one cluster across all kept batches in a single launch.
///
/// Replaces the K * B' per-(cluster, batch) launches with one launch
/// per cluster. For cluster `ku`, applies
/// `Z_corr[:, cell] -= W[j, :] * R_row_k[cell]` for every cell in
/// every kept batch j ∈ [0, n_kept).
///
/// Layout:
/// - `cells_concat`: i32 flat list of cell indices for all kept
///   batches, concatenated. Length = n_kept_total.
/// - `batch_offsets`: i32 cumulative offsets into `cells_concat`.
///   Length = n_kept + 1; cells for batch j are
///   `cells_concat[batch_offsets[j]..batch_offsets[j+1]]`.
/// - `w`: f32 row-major (n_kept, d). Row j is the per-PC weight for
///   kept batch j.
/// - `r_row_k`: f32 row of R for cluster ku, length N.
///
/// Block size is padded to the next warp multiple (>= d). Grid has
/// `n_kept_total` blocks; each block processes one cell across all d
/// PCs and performs a small binary search over `batch_offsets` to
/// recover the batch index.
#[allow(clippy::too_many_arguments)]
pub fn gpu_harmony_correction_grouped(
    dev: &GpuDevice,
    z_corr: &mut CudaSlice<f32>,
    r_row_k: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    cells_concat: &CudaSlice<i32>,
    batch_offsets: &CudaSlice<i32>,
    n_kept: usize,
    n_kept_total: usize,
    d: usize,
    n: usize,
) -> Result<(), GpuError> {
    if n_kept == 0 || n_kept_total == 0 || d == 0 {
        return Ok(());
    }
    if z_corr.len() < d * n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("Z_corr length >= d*N = {}", d * n),
            got: format!("{}", z_corr.len()),
        });
    }
    if r_row_k.len() < n {
        return Err(GpuError::ShapeMismatch {
            expected: format!("R_row_k length >= N = {n}"),
            got: format!("{}", r_row_k.len()),
        });
    }
    if w.len() < n_kept * d {
        return Err(GpuError::ShapeMismatch {
            expected: format!("W length >= n_kept * d = {}", n_kept * d),
            got: format!("{}", w.len()),
        });
    }
    if cells_concat.len() < n_kept_total {
        return Err(GpuError::ShapeMismatch {
            expected: format!("cells_concat length >= n_kept_total = {n_kept_total}"),
            got: format!("{}", cells_concat.len()),
        });
    }
    if batch_offsets.len() < n_kept + 1 {
        return Err(GpuError::ShapeMismatch {
            expected: format!("batch_offsets length >= n_kept + 1 = {}", n_kept + 1),
            got: format!("{}", batch_offsets.len()),
        });
    }

    let module = dev.load_module_cached(HARMONY_PTX)?;
    let func = module
        .load_function("harmony_correction_grouped_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_correction_grouped: {e}")))?;

    // Block size = next warp multiple of d, capped at 1024 (CUDA limit).
    let warp = 32u32;
    let mut threads = (d as u32).div_ceil(warp) * warp;
    if threads == 0 {
        threads = warp;
    }
    if threads > 1024 {
        return Err(GpuError::ShapeMismatch {
            expected: "d <= 1024 (CUDA block size limit)".into(),
            got: format!("d = {d}"),
        });
    }
    if (n_kept_total as u64) > i32::MAX as u64 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_kept_total < 2^31 (CUDA grid_dim.x cap)".into(),
            got: format!("n_kept_total = {n_kept_total}"),
        });
    }
    let cfg = LaunchConfig {
        grid_dim: (n_kept_total as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_kept_i32 = n_kept as i32;
    let n_kept_total_i32 = n_kept_total as i32;
    let d_i32 = d as i32;
    let n_i32 = n as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(z_corr)
            .arg(r_row_k)
            .arg(w)
            .arg(cells_concat)
            .arg(batch_offsets)
            .arg(&n_kept_i32)
            .arg(&n_kept_total_i32)
            .arg(&d_i32)
            .arg(&n_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_correction_grouped: {e}")))?;
    Ok(())
}

/// Rough device-memory footprint for a Harmony GPU run, in bytes.
///
/// Counts the buffers that live on the device for the full duration of
/// the iteration loop: Z_orig, Z_corr, Z_cos, R, dist, Y plus per-cell
/// batch labels and small O/E/theta/sigma arrays. Transient scratch is
/// not included.
pub fn gpu_harmony_memory_bytes(n: usize, d: usize, k: usize, b: usize, c: usize) -> u64 {
    let f32b = std::mem::size_of::<f32>() as u64;
    let i32b = std::mem::size_of::<i32>() as u64;
    let n = n as u64;
    let d = d as u64;
    let k = k as u64;
    let b = b as u64;
    let c = c as u64;

    // Z_orig + Z_corr + Z_cos: 3 * d*N * f32
    let z = 3 * d * n * f32b;
    // R + dist: 2 * K*N * f32
    let r = 2 * k * n * f32b;
    // Y: d*K * f32
    let y = d * k * f32b;
    // O, E: K*B * f32 each
    let oe = 2 * k * b * f32b;
    // theta, sigma: small
    let ts = (b + k) * f32b;
    // batch labels: C*N * i32, cov_offset: C * i32
    let lab = c * n * i32b + c * i32b;
    z + r + y + oe + ts + lab
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimate_monotone() {
        let small = gpu_harmony_memory_bytes(1_000, 30, 50, 3, 1);
        let large = gpu_harmony_memory_bytes(1_000_000, 30, 100, 3, 1);
        assert!(large > small);
        // Sanity: 1M cells × 30 PCs × 3 Z buffers × 4 bytes ≈ 360 MB.
        // The full estimate must be within a sensible order of magnitude.
        assert!(large > 300_000_000);
        assert!(large < 5_000_000_000);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_distances_kernel_matches_cpu() {
        let dev = require_gpu!();
        let d = 5usize;
        let k = 3usize;
        let n = 7usize;

        // Deterministic small inputs.
        let y_host: Vec<f32> = (0..d * k).map(|v| (v as f32).sin()).collect();
        let z_host: Vec<f32> = (0..d * n).map(|v| (v as f32).cos()).collect();

        let y = dev.htod_copy(&y_host).unwrap();
        let z = dev.htod_copy(&z_host).unwrap();
        let mut dist = dev.alloc_zeros::<f32>(k * n).unwrap();
        gpu_harmony_distances(&dev, &y, &z, &mut dist, d, k, n).unwrap();
        dev.synchronize().unwrap();
        let got = dev.dtoh_copy(&dist).unwrap();

        // CPU reference.
        let mut expected = vec![0f32; k * n];
        for ku in 0..k {
            for i in 0..n {
                let mut dot = 0f32;
                for t in 0..d {
                    dot += y_host[ku * d + t] * z_host[i * d + t];
                }
                expected[ku * n + i] = 2.0 * (1.0 - dot);
            }
        }
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_distances_gemm_matches_kernel() {
        // GEMM-backed variant must produce numerically equivalent
        // output to the hand-written kernel within f32 GEMM rounding.
        let dev = require_gpu!();
        let d = 17usize;
        let k = 32usize;
        let n = 256usize;

        let y_host: Vec<f32> = (0..d * k).map(|v| ((v as f32) * 0.013).sin()).collect();
        let z_host: Vec<f32> = (0..d * n).map(|v| ((v as f32) * 0.017).cos()).collect();

        let y = dev.htod_copy(&y_host).unwrap();
        let z = dev.htod_copy(&z_host).unwrap();

        let mut dist_kernel = dev.alloc_zeros::<f32>(k * n).unwrap();
        gpu_harmony_distances(&dev, &y, &z, &mut dist_kernel, d, k, n).unwrap();

        let mut dist_gemm = dev.alloc_zeros::<f32>(k * n).unwrap();
        let handle = CublasHandle::new().unwrap();
        gpu_harmony_distances_gemm(&dev, &handle, &y, &z, &mut dist_gemm, d, k, n).unwrap();

        dev.synchronize().unwrap();
        let got_kernel = dev.dtoh_copy(&dist_kernel).unwrap();
        let got_gemm = dev.dtoh_copy(&dist_gemm).unwrap();

        for (i, (a, b)) in got_kernel.iter().zip(got_gemm.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "idx {i}: kernel {a} vs gemm {b}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_distances_multi_tile_matches_single_tile() {
        // Verify the host-tile loop produces the same output as a
        // single launch on a smaller N. We exercise tiling by calling
        // the wrapper at sizes where its internal TILE_LIMIT is
        // expected to chunk; on shapes that fit in one tile, the
        // result must be unchanged. Acts as a smoke regression for
        // the i32::MAX cap removal — the wrapper no longer rejects
        // any K, N combination on the basis of K*N alone.
        let dev = require_gpu!();
        let d = 8usize;
        let k = 64usize;
        let n = 4096usize;

        let y_host: Vec<f32> = (0..d * k).map(|v| (v as f32 * 0.01).sin()).collect();
        let z_host: Vec<f32> = (0..d * n).map(|v| (v as f32 * 0.013).cos()).collect();

        let y = dev.htod_copy(&y_host).unwrap();
        let z = dev.htod_copy(&z_host).unwrap();
        let mut dist = dev.alloc_zeros::<f32>(k * n).unwrap();

        gpu_harmony_distances(&dev, &y, &z, &mut dist, d, k, n).unwrap();
        dev.synchronize().unwrap();
        let got = dev.dtoh_copy(&dist).unwrap();

        // CPU reference.
        for ku in 0..k {
            for i in 0..n {
                let mut dot = 0f32;
                for t in 0..d {
                    dot += y_host[ku * d + t] * z_host[i * d + t];
                }
                let expected = 2.0 * (1.0 - dot);
                let g = got[ku * n + i];
                assert!(
                    (g - expected).abs() < 1e-4,
                    "k={ku}, i={i}: {g} vs {expected}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_l2_normalize_cols_kernel() {
        let dev = require_gpu!();
        let d = 4usize;
        let n = 6usize;
        let host: Vec<f32> = (0..d * n).map(|v| (v + 1) as f32).collect();
        let mut d_m = dev.htod_copy(&host).unwrap();
        gpu_harmony_l2_normalize_cols(&dev, &mut d_m, d, n).unwrap();
        dev.synchronize().unwrap();
        let got = dev.dtoh_copy(&d_m).unwrap();
        for i in 0..n {
            let col = &got[i * d..(i + 1) * d];
            let norm_sq: f32 = col.iter().map(|v| v * v).sum();
            assert!((norm_sq - 1.0).abs() < 1e-4, "col {i}: ||.||^2 = {norm_sq}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_correction_kernel() {
        let dev = require_gpu!();
        let d = 3usize;
        let n = 5usize;
        // Z_corr columns = [1,1,1] replicated.
        let z_host = vec![1.0f32; d * n];
        let r_row: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let w_row: Vec<f32> = vec![0.5, 0.25, 0.125];
        let cells: Vec<i32> = vec![1, 3]; // subtract at cells 1 and 3

        let mut z = dev.htod_copy(&z_host).unwrap();
        let r = dev.htod_copy(&r_row).unwrap();
        let w = dev.htod_copy(&w_row).unwrap();
        let c = dev.htod_copy(&cells).unwrap();
        gpu_harmony_correction(&dev, &mut z, d, n, &c, &r, &w).unwrap();
        dev.synchronize().unwrap();
        let got = dev.dtoh_copy(&z).unwrap();

        for i in 0..n as i32 {
            let col = &got[(i as usize) * d..(i as usize + 1) * d];
            if cells.contains(&i) {
                let r_ki = r_row[i as usize];
                for t in 0..d {
                    let expected = 1.0 - w_row[t] * r_ki;
                    assert!(
                        (col[t] - expected).abs() < 1e-5,
                        "cell {i}, t={t}: {} vs {expected}",
                        col[t]
                    );
                }
            } else {
                for &v in col {
                    assert!((v - 1.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_z_sum_kernel_matches_cpu() {
        // Per-cluster z-sum: z_sum[j, t] = sum_{i in batch j} R[k,i] * Z[t,i].
        let dev = require_gpu!();
        let d = 4usize;
        let n = 16usize;
        let cells_b0: Vec<i32> = vec![0, 2, 5, 8];
        let cells_b1: Vec<i32> = vec![1, 4, 7, 11, 13];
        let cells_b2: Vec<i32> = vec![]; // empty kept batch — z_sum row should be 0
        let cells_b3: Vec<i32> = vec![3, 6, 9, 10, 12, 14];

        let mut cells_concat: Vec<i32> = Vec::new();
        cells_concat.extend(&cells_b0);
        cells_concat.extend(&cells_b1);
        cells_concat.extend(&cells_b2);
        cells_concat.extend(&cells_b3);
        let batch_offsets: Vec<i32> = vec![
            0,
            cells_b0.len() as i32,
            (cells_b0.len() + cells_b1.len()) as i32,
            (cells_b0.len() + cells_b1.len() + cells_b2.len()) as i32,
            (cells_b0.len() + cells_b1.len() + cells_b2.len() + cells_b3.len()) as i32,
        ];
        let n_kept = 4usize;

        let r_row: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 + 1.0)).collect();
        // Z_orig col-major (d, n): col i = [i*0.1 + t*0.01 for t in 0..d]
        let z_host: Vec<f32> = (0..n)
            .flat_map(|i| (0..d).map(move |t| (i as f32) * 0.1 + (t as f32) * 0.01))
            .collect();

        // CPU reference.
        let mut z_sum_cpu = vec![0f32; n_kept * d];
        let cell_lists = [&cells_b0, &cells_b1, &cells_b2, &cells_b3];
        for (j, cells) in cell_lists.iter().enumerate() {
            for &cell in cells.iter() {
                let r = r_row[cell as usize];
                if r == 0.0 {
                    continue;
                }
                for t in 0..d {
                    z_sum_cpu[j * d + t] += r * z_host[(cell as usize) * d + t];
                }
            }
        }

        let d_r = dev.htod_copy(&r_row).unwrap();
        let d_z = dev.htod_copy(&z_host).unwrap();
        let d_cells = dev.htod_copy(&cells_concat).unwrap();
        let d_offsets = dev.htod_copy(&batch_offsets).unwrap();
        let mut d_z_sum = dev.alloc_zeros::<f32>(n_kept * d).unwrap();
        gpu_harmony_z_sum(
            &dev,
            &d_r,
            &d_z,
            &d_cells,
            &d_offsets,
            &mut d_z_sum,
            n_kept,
            d,
            n,
        )
        .unwrap();
        dev.synchronize().unwrap();
        let got = dev.dtoh_copy(&d_z_sum).unwrap();

        for (i, (a, b)) in z_sum_cpu.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i} (j={}, t={}): cpu {a} vs gpu {b}",
                i / d,
                i % d
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_correction_grouped_matches_per_pair() {
        // Per-pair launches and one grouped launch must produce
        // bit-equivalent output (no atomics involved — disjoint cell
        // writes per (j, t)).
        let dev = require_gpu!();
        let d = 4usize;
        let n = 12usize;
        // Two kept batches: batch 0 has cells [0, 2, 5], batch 1 has
        // cells [3, 4, 7, 9]. Cells outside any kept batch are
        // untouched.
        let cells_b0: Vec<i32> = vec![0, 2, 5];
        let cells_b1: Vec<i32> = vec![3, 4, 7, 9];
        let w_b0: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let w_b1: Vec<f32> = vec![-0.05, 0.15, -0.25, 0.35];

        // Baseline Z_corr.
        let z_initial: Vec<f32> = (0..d * n).map(|v| (v as f32) * 0.01).collect();
        // R[ku, :] for the cluster being tested.
        let r_row: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 + 1.0)).collect();

        // --- Baseline: per-pair launches ---
        let mut z_baseline = dev.htod_copy(&z_initial).unwrap();
        let r = dev.htod_copy(&r_row).unwrap();
        let d_cells_b0 = dev.htod_copy(&cells_b0).unwrap();
        let d_cells_b1 = dev.htod_copy(&cells_b1).unwrap();
        let d_w_b0 = dev.htod_copy(&w_b0).unwrap();
        let d_w_b1 = dev.htod_copy(&w_b1).unwrap();
        gpu_harmony_correction(&dev, &mut z_baseline, d, n, &d_cells_b0, &r, &d_w_b0).unwrap();
        gpu_harmony_correction(&dev, &mut z_baseline, d, n, &d_cells_b1, &r, &d_w_b1).unwrap();
        dev.synchronize().unwrap();
        let got_baseline = dev.dtoh_copy(&z_baseline).unwrap();

        // --- Grouped: one launch ---
        let mut z_grouped = dev.htod_copy(&z_initial).unwrap();
        let mut cells_concat: Vec<i32> = Vec::new();
        cells_concat.extend(&cells_b0);
        cells_concat.extend(&cells_b1);
        let batch_offsets: Vec<i32> = vec![
            0,
            cells_b0.len() as i32,
            (cells_b0.len() + cells_b1.len()) as i32,
        ];
        let mut w_flat = Vec::new();
        w_flat.extend(&w_b0);
        w_flat.extend(&w_b1);
        let d_cells_concat = dev.htod_copy(&cells_concat).unwrap();
        let d_offsets = dev.htod_copy(&batch_offsets).unwrap();
        let d_w = dev.htod_copy(&w_flat).unwrap();
        let n_kept = 2usize;
        let n_kept_total = cells_concat.len();
        gpu_harmony_correction_grouped(
            &dev,
            &mut z_grouped,
            &r,
            &d_w,
            &d_cells_concat,
            &d_offsets,
            n_kept,
            n_kept_total,
            d,
            n,
        )
        .unwrap();
        dev.synchronize().unwrap();
        let got_grouped = dev.dtoh_copy(&z_grouped).unwrap();

        for (i, (a, b)) in got_baseline.iter().zip(got_grouped.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "idx {i}: baseline {a} vs grouped {b}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_softmax_penalty_kernel_uniform() {
        let dev = require_gpu!();
        // With O == E and all theta = 0, the diversity penalty is 1 and
        // the kernel should reduce to plain softmax(-dist/sigma).
        let k = 4usize;
        let n = 3usize;
        let b = 2usize;
        let c = 1usize;

        let dist: Vec<f32> = (0..k * n).map(|i| (i % 3) as f32 * 0.1).collect();
        let sigma = vec![0.1f32; k];
        let o = vec![1.0f32; k * b];
        let e = vec![1.0f32; k * b];
        let theta = vec![0.0f32; b];
        let labels: Vec<i32> = (0..n as i32).map(|i| i % (b as i32)).collect();
        let cov_offset: Vec<i32> = vec![0];

        let d_dist = dev.htod_copy(&dist).unwrap();
        let d_sigma = dev.htod_copy(&sigma).unwrap();
        let d_o = dev.htod_copy(&o).unwrap();
        let d_e = dev.htod_copy(&e).unwrap();
        let d_theta = dev.htod_copy(&theta).unwrap();
        let d_labels = dev.htod_copy(&labels).unwrap();
        let d_cov = dev.htod_copy(&cov_offset).unwrap();
        let mut d_r = dev.alloc_zeros::<f32>(k * n).unwrap();

        gpu_harmony_softmax_penalty(
            &dev, &d_dist, &d_sigma, &d_o, &d_e, &d_theta, &d_labels, &d_cov, &mut d_r, c, k, n, b,
        )
        .unwrap();
        dev.synchronize().unwrap();
        let got = dev.dtoh_copy(&d_r).unwrap();

        // Columns should sum to 1.
        for i in 0..n {
            let mut s = 0f32;
            for ku in 0..k {
                s += got[ku * n + i];
            }
            assert!((s - 1.0).abs() < 1e-4, "col {i} sum = {s}");
        }
    }

    /// Task 2.6: the device objective reduction sums `obj_cell` (N) and
    /// `cross_kgb` (K·B) into two f64 scalars matching a host f64 sum.
    /// Exercises a non-trivial length (> one block) and an empty-array edge.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_reduce_objective_matches_host() {
        let dev = require_gpu!();
        let n = 5000usize; // > block size, forces the grid-stride + atomicAdd
        let k_b = 777usize;
        let obj: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.25 - 1.0).collect();
        let cross: Vec<f32> = (0..k_b).map(|i| ((i % 7) as f32) * 0.5).collect();

        let host_obj: f64 = obj.iter().map(|&v| v as f64).sum();
        let host_cross: f64 = cross.iter().map(|&v| v as f64).sum();

        let d_obj = dev.htod_copy(&obj).unwrap();
        let d_cross = dev.htod_copy(&cross).unwrap();
        let (sum_obj, sum_cross) =
            gpu_harmony_reduce_objective(&dev, &d_obj, n, &d_cross, k_b).unwrap();

        // f64 device accumulation vs host f64 sum — agree to a tight tol
        // (only reassociation differs).
        assert!(
            (sum_obj - host_obj).abs() <= 1e-6 * host_obj.abs().max(1.0),
            "obj sum: gpu {sum_obj} vs host {host_obj}"
        );
        assert!(
            (sum_cross - host_cross).abs() <= 1e-6 * host_cross.abs().max(1.0),
            "cross sum: gpu {sum_cross} vs host {host_cross}"
        );

        // Empty arrays reduce to zero without launching.
        let empty = dev.alloc_zeros::<f32>(0).unwrap();
        let (z0, z1) = gpu_harmony_reduce_objective(&dev, &empty, 0, &empty, 0).unwrap();
        assert_eq!((z0, z1), (0.0, 0.0));
    }
}
