//! GPU CSR → dense conversion with optional HVG (gene subset) projection.
//!
//! Wraps the `sparse_to_dense_kernel` CUDA kernel. The kernel uses warp-
//! cooperative row scatter: each warp (32 threads) handles one row's
//! non-zeros in parallel, with optional column remapping for HVG projection.

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_decode::GpuCsr;

/// Compiled PTX for the sparse-to-dense kernel (produced by build.rs via nvcc --ptx).
const SPARSE_DENSE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sparse_dense.ptx"));

/// Convert a GPU-resident CSR matrix to a dense row-major matrix, allocating
/// a fresh zero-initialised output buffer.
///
/// # Arguments
///
/// * `device` — GPU device handle.
/// * `gpu_csr` — GPU-resident CSR matrix (i64 indptr, i32 indices, f32 data).
/// * `hvg_map` — Optional gene projection map on device. `hvg_map[original_col]`
///   gives the output column index, or `0xFFFFFFFF` to skip. If `None`,
///   identity mapping is used (all columns, `n_output_cols == n_cols`).
/// * `n_output_cols` — Number of columns in the dense output. Must equal
///   `gpu_csr.shape.1` when `hvg_map` is `None`, or the number of selected
///   genes when projecting.
///
/// # Returns
///
/// A `CudaSlice<f32>` of size `n_rows × n_output_cols`, row-major, zero-filled
/// for absent entries.
///
/// Thin wrapper around [`sparse_to_dense_gpu_into`] for one-shot callers that
/// don't reuse a scratch buffer. Loop callers (e.g. `accumulate_gram`) should
/// hoist the buffer and call [`sparse_to_dense_gpu_into`] directly.
pub fn sparse_to_dense_gpu(
    device: &GpuDevice,
    gpu_csr: &GpuCsr,
    hvg_map: Option<&CudaSlice<u32>>,
    n_output_cols: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let (n_rows, _n_cols) = gpu_csr.shape;

    if n_rows == 0 || n_output_cols == 0 {
        return device.alloc_zeros::<f32>(0);
    }

    let mut d_output = device.alloc_zeros::<f32>(n_rows * n_output_cols)?;
    sparse_to_dense_gpu_into(device, gpu_csr, hvg_map, n_output_cols, &mut d_output)?;
    Ok(d_output)
}

/// Convert a GPU-resident CSR matrix to a dense row-major matrix, writing into
/// a caller-supplied scratch buffer instead of allocating.
///
/// # Buffer contract
///
/// `scratch` must satisfy `scratch.len() >= n_rows * n_output_cols`. The
/// kernel only writes positions corresponding to nonzeros (and, with
/// `hvg_map`, mapped output columns); empty rows and unmapped columns are
/// left **unchanged**. Callers who reuse the same scratch across multiple
/// CSRs must therefore zero the leading `n_rows * n_output_cols` region
/// before each call to avoid carrying over stale values from a previous
/// shard. `cudarc::driver::safe::CudaStream::memset_zeros` on a
/// `slice_mut(0..n_rows * n_output_cols)` view is the cheapest way to do
/// this.
///
/// Allocator-allocated buffers (e.g. via `alloc_zeros`) are zero-filled by
/// construction; the first call after allocation does not need an explicit
/// zero pass.
///
/// # Arguments
///
/// Same as [`sparse_to_dense_gpu`], plus `scratch` — the caller-owned output
/// buffer.
pub fn sparse_to_dense_gpu_into(
    device: &GpuDevice,
    gpu_csr: &GpuCsr,
    hvg_map: Option<&CudaSlice<u32>>,
    n_output_cols: usize,
    scratch: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let (n_rows, _n_cols) = gpu_csr.shape;

    if n_rows == 0 || n_output_cols == 0 {
        return Ok(());
    }

    let needed = n_rows * n_output_cols;
    if scratch.len() < needed {
        return Err(GpuError::KernelLaunchFailed(format!(
            "sparse_to_dense_gpu_into: scratch buffer too small ({} < {})",
            scratch.len(),
            needed
        )));
    }

    // Load PTX module (cached) and get kernel function
    let module = device.load_module_cached(SPARSE_DENSE_PTX)?;
    let kernel = module
        .load_function("sparse_to_dense_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load sparse_to_dense_kernel: {e}")))?;

    let n_rows_i32 = n_rows as i32;
    let n_output_cols_i32 = n_output_cols as i32;

    // Launch config: one warp (32 threads) per row.
    // CUDA block size = 256 threads = 8 warps per block.
    let threads_per_block: u32 = 256;
    let warps_per_block = threads_per_block / 32;
    let grid_dim = (n_rows as u32).div_ceil(warps_per_block);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Build kernel argument for hvg_map (null pointer if None).
    // cudarc expects all arguments to be pushed in order; for a nullable pointer
    // we pass either the device slice or a raw null u64.
    match hvg_map {
        Some(hmap) => unsafe {
            device
                .stream()
                .launch_builder(&kernel)
                .arg(&gpu_csr.indptr)
                .arg(&gpu_csr.indices)
                .arg(&gpu_csr.data)
                .arg(scratch)
                .arg(hmap)
                .arg(&n_rows_i32)
                .arg(&n_output_cols_i32)
                .launch(cfg)
        },
        None => {
            // Pass a null device pointer for hvg_map.
            let null_ptr: u64 = 0;
            unsafe {
                device
                    .stream()
                    .launch_builder(&kernel)
                    .arg(&gpu_csr.indptr)
                    .arg(&gpu_csr.indices)
                    .arg(&gpu_csr.data)
                    .arg(scratch)
                    .arg(&null_ptr)
                    .arg(&n_rows_i32)
                    .arg(&n_output_cols_i32)
                    .launch(cfg)
            }
        }
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("sparse_to_dense_kernel: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_decode::decode_shard_gpu;
    use crate::test_utils::build_test_shard;
    use scx_codec::{CodecId, ValueEncoding};

    /// CPU reference: scatter CSR arrays into a dense matrix.
    fn cpu_sparse_to_dense(
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
    ) -> Vec<f32> {
        let mut dense = vec![0.0f32; n_rows * n_cols];
        for row in 0..n_rows {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for nz in start..end {
                let col = indices[nz] as usize;
                if col < n_cols {
                    dense[row * n_cols + col] = data[nz];
                }
            }
        }
        dense
    }

    /// CPU reference: scatter CSR arrays into a dense matrix with HVG projection.
    fn cpu_sparse_to_dense_hvg(
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        n_rows: usize,
        hvg_map: &[u32],
        n_output_cols: usize,
    ) -> Vec<f32> {
        let mut dense = vec![0.0f32; n_rows * n_output_cols];
        for row in 0..n_rows {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for nz in start..end {
                let col = indices[nz] as usize;
                if col < hvg_map.len() {
                    let mapped = hvg_map[col];
                    if mapped != 0xFFFFFFFF && (mapped as usize) < n_output_cols {
                        dense[row * n_output_cols + mapped as usize] = data[nz];
                    }
                }
            }
        }
        dense
    }

    #[test]
    fn test_sparse_to_dense_gpu_matches_cpu() {
        let dev = require_gpu!();

        // Build a 5-row, 100-col CSR shard
        let n_cols: u32 = 100;
        let indptr = vec![0u64, 3, 5, 5, 8, 12];
        let indices = vec![
            0u32, 10, 50, // row 0
            5, 15, // row 1
            // row 2 empty
            1, 2, 3, // row 3
            20, 40, 60, 80, // row 4
        ];
        let values_u16: Vec<u16> = (1..=12).collect();
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        // GPU decode to GpuCsr
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        assert_eq!(gpu_csr.shape, (5, 100));

        // GPU sparse → dense (no HVG)
        let d_dense = sparse_to_dense_gpu(&dev, &gpu_csr, None, n_cols as usize).unwrap();
        let gpu_dense = dev.dtoh_copy(&d_dense).unwrap();

        // CPU reference
        let cpu_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let cpu_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
        let cpu_data: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let cpu_dense = cpu_sparse_to_dense(&cpu_indptr, &cpu_indices, &cpu_data, 5, 100);

        assert_eq!(gpu_dense.len(), cpu_dense.len(), "output size mismatch");
        assert_eq!(
            gpu_dense, cpu_dense,
            "GPU dense output must match CPU reference"
        );
    }

    #[test]
    fn test_sparse_to_dense_with_hvg() {
        let dev = require_gpu!();

        // Same 5-row, 100-col CSR
        let n_cols: u32 = 100;
        let indptr = vec![0u64, 3, 5, 5, 8, 12];
        let indices = vec![
            0u32, 10, 50, // row 0
            5, 15, // row 1
            // row 2 empty
            1, 2, 3, // row 3
            20, 40, 60, 80, // row 4
        ];
        let values_u16: Vec<u16> = (1..=12).collect();
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        // GPU decode to GpuCsr
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();

        // HVG subset: keep only columns 0, 2, 10, 50, 80 → output cols 0..4
        // Build a mapping: hvg_map[col] = output_col or 0xFFFFFFFF
        let selected_cols: Vec<u32> = vec![0, 2, 10, 50, 80];
        let n_output_cols = selected_cols.len();
        let mut hvg_map = vec![0xFFFFFFFFu32; n_cols as usize];
        for (out_col, &orig_col) in selected_cols.iter().enumerate() {
            hvg_map[orig_col as usize] = out_col as u32;
        }
        let d_hvg_map = dev.htod_copy(&hvg_map).unwrap();

        // GPU sparse → dense with HVG
        let d_dense = sparse_to_dense_gpu(&dev, &gpu_csr, Some(&d_hvg_map), n_output_cols).unwrap();
        let gpu_dense = dev.dtoh_copy(&d_dense).unwrap();

        // CPU reference
        let cpu_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let cpu_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
        let cpu_data: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
        let cpu_dense = cpu_sparse_to_dense_hvg(
            &cpu_indptr,
            &cpu_indices,
            &cpu_data,
            5,
            &hvg_map,
            n_output_cols,
        );

        assert_eq!(
            gpu_dense.len(),
            5 * n_output_cols,
            "output size should be n_rows × n_output_cols"
        );
        assert_eq!(gpu_dense, cpu_dense, "HVG-projected dense must match CPU");

        // Verify specific values:
        // Row 0 has col=0 (val=1.0), col=10 (val=2.0), col=50 (val=3.0)
        // Mapped: col 0→out 0, col 10→out 2, col 50→out 3
        assert_eq!(gpu_dense[0 * n_output_cols + 0], 1.0, "row0 col0");
        assert_eq!(gpu_dense[0 * n_output_cols + 2], 2.0, "row0 col10→out2");
        assert_eq!(gpu_dense[0 * n_output_cols + 3], 3.0, "row0 col50→out3");
        // Unmapped columns should be zero
        assert_eq!(gpu_dense[0 * n_output_cols + 1], 0.0, "row0 unmapped col2");
    }
}
