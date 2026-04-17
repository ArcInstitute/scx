//! GPU kernels for Harmony2 batch integration.
//!
//! Exposes four primitive operations as host-side functions that take
//! device-resident `CudaSlice<f32>` buffers. Callers own the memory and
//! the orchestration loop — see [`scx_accel::harmony_integrate_gpu`].
//!
//! Layout contracts (match `scx-accel/src/harmony.rs` CPU code):
//! - `Z_*` matrices: `(d x N)` column-major (`Z[t, i] = flat[i*d + t]`).
//! - `Y`: `(d x K)` column-major.
//! - `R`, `dist`, `O`, `E`: row-major (row-major stride = last dim).

use cudarc::driver::safe::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::device::GpuDevice;
use crate::error::GpuError;

/// PTX for all Harmony kernels (compiled from `scx-gpu/kernels/harmony.cu`).
const HARMONY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/harmony.ptx"));

/// Compute `dist[k, i] = 2 * (1 - dot(Y[:, k], Z_cos[:, i]))` on GPU.
///
/// * `y` — `(d x K)` col-major.
/// * `z_cos` — `(d x N)` col-major.
/// * `dist_out` — `(K x N)` row-major, must be pre-allocated.
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

    let total: u64 = (k as u64) * (n as u64);
    // The kernel uses int32 for the flat index; enforce that bound so callers
    // don't silently overflow on truly enormous workloads.
    if total >= (i32::MAX as u64) {
        return Err(GpuError::ShapeMismatch {
            expected: "K*N < 2^31".into(),
            got: format!("K*N = {total}"),
        });
    }

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let d_i32 = d as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(y)
            .arg(z_cos)
            .arg(dist_out)
            .arg(&d_i32)
            .arg(&k_i32)
            .arg(&n_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("harmony_distances: {e}")))?;
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
}
