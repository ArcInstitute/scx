//! GPU type-cast kernels for on-device u32→i32 and u32→f32 conversion.
//!
//! These eliminate the GPU→CPU→GPU round-trips that would otherwise be
//! needed when the decode kernels produce `u32` but the CSR output
//! requires `i32` (indices) and `f32` (values).

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// Compiled PTX for cast kernels (produced by build.rs via nvcc --ptx).
const CAST_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cast.ptx"));

/// Cast a GPU buffer of `u32` to `i32` on-device.
///
/// Safe for all values < 2^31 (always true for column indices in SCX).
pub fn cast_u32_to_i32_gpu(
    dev: &GpuDevice,
    input: &CudaSlice<u32>,
) -> Result<CudaSlice<i32>, GpuError> {
    let n = input.len();
    if n == 0 {
        return dev.alloc_zeros::<i32>(0);
    }

    let module = dev.load_module_cached(CAST_PTX)?;
    let kernel = module
        .load_function("cast_u32_to_i32")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load cast_u32_to_i32: {e}")))?;

    let mut output = dev.alloc_zeros::<i32>(n)?;
    let n_u32 = n as u32;

    let threads: u32 = 256;
    let grid = (n as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(input)
            .arg(&mut output)
            .arg(&n_u32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("cast_u32_to_i32: {e}")))?;

    Ok(output)
}

/// Cast a GPU buffer of `u32` to `f32` on-device.
///
/// Exact for all values ≤ 2^24 (always true for UMI counts in SCX).
pub fn cast_u32_to_f32_gpu(
    dev: &GpuDevice,
    input: &CudaSlice<u32>,
) -> Result<CudaSlice<f32>, GpuError> {
    let n = input.len();
    if n == 0 {
        return dev.alloc_zeros::<f32>(0);
    }

    let module = dev.load_module_cached(CAST_PTX)?;
    let kernel = module
        .load_function("cast_u32_to_f32")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load cast_u32_to_f32: {e}")))?;

    let mut output = dev.alloc_zeros::<f32>(n)?;
    let n_u32 = n as u32;

    let threads: u32 = 256;
    let grid = (n as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(input)
            .arg(&mut output)
            .arg(&n_u32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("cast_u32_to_f32: {e}")))?;

    Ok(output)
}
